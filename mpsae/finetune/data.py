"""Raw-image loaders with replayable augmentation and checkpoint-specific features."""
from pathlib import Path

from ..io import canonical_hash, read_json


class EpochImages:
    def __init__(self, dataset, transform, seed, epoch=0):
        self.dataset, self.transform, self.seed, self.epoch = dataset, transform, seed, epoch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        epoch = self.epoch
        if isinstance(index, tuple):
            epoch, index = index
        index = int(index)
        image, label = self.dataset[index]
        # Each image has the same augmentation under any worker count/resume.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed + epoch * len(self.dataset) + index)
            return self.transform(image), label


class EpochBatchSampler:
    """Microbatches never cross an optimizer-batch boundary, even on the tail."""
    def __init__(self, rows, seed, batch_size, microbatch, drop_last):
        self.rows, self.seed, self.batch_size = rows, seed, batch_size
        self.microbatch = min(microbatch or batch_size, batch_size)
        self.drop_last, self.epoch, self.skip = drop_last, 0, 0

    def sizes(self):
        for start in range(self.skip, self.rows, self.batch_size):
            size = min(self.batch_size, self.rows - start)
            if size < self.batch_size and self.drop_last:
                break
            yield size

    def __iter__(self):
        import torch
        order = torch.randperm(self.rows, generator=torch.Generator().manual_seed(
            self.seed + self.epoch)).numpy()
        start = self.skip
        for size in self.sizes():
            for offset in range(0, size, self.microbatch):
                yield [(self.epoch, int(i)) for i in order[
                    start + offset:start + min(offset + self.microbatch, size)]]
            start += size


class TrainingLoader:
    """Persistent workers; seeded epoch indices also work with spawn and resumes."""
    def __init__(self, dataset, transform, *, seed, batch_size, microbatch=0,
                 workers=0, drop_last=False, prefetch_factor=2):
        import torch
        from torch.utils.data import DataLoader
        self.sampler = EpochBatchSampler(len(dataset), seed, batch_size, microbatch, drop_last)
        options = dict(num_workers=workers, pin_memory=torch.cuda.is_available(),
                       generator=torch.Generator().manual_seed(seed))
        if workers:
            options.update(persistent_workers=True, prefetch_factor=prefetch_factor)
        self.loader = DataLoader(EpochImages(dataset, transform, seed),
                                 batch_sampler=self.sampler, **options)

    def batches(self, epoch, skip=0):
        self.sampler.epoch, self.sampler.skip = epoch, skip
        iterator = iter(self.loader)
        for size in self.sampler.sizes():
            chunks = [next(iterator) for _ in range(-(-size // self.sampler.microbatch))]
            yield chunks

    def close(self):
        # DataLoader has no public close. Release persistent workers promptly on
        # interrupts too, before moving on to another backbone or feature loader.
        iterator = getattr(self.loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            self.loader._iterator = None


def loader(dataset, transform, *, seed, epoch, batch_size, workers, skip=0, training=False, drop_last=None):
    import torch
    from torch.utils.data import DataLoader, Subset
    generator = torch.Generator().manual_seed(seed + epoch)
    # Compact indices avoid millions of Python integers (and worker copies).
    # Keep the exact torch permutation and skip offset used by old checkpoints.
    order = torch.randperm(len(dataset), generator=generator).numpy() if training else range(len(dataset))
    rows = EpochImages(dataset, transform, seed, epoch)
    return DataLoader(Subset(rows, order[skip:]), batch_size=batch_size, shuffle=False,
                      drop_last=training if drop_last is None else drop_last,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed + epoch))


def image_datasets(image_root):
    from torchvision.datasets import ImageFolder
    from ..data.imagenet import imagenet_folder_stats
    root = Path(image_root)
    datasets, identity = {}, {}
    for split, expected in (("train", 1_281_167), ("val", 50_000)):
        stats = imagenet_folder_stats(root / split)
        if stats["images"] != expected or stats["classes"] != 1000:
            raise RuntimeError(f"{root / split}: expected {expected} images in 1000 classes")
        datasets[split] = ImageFolder(str(root / split))
        if len(datasets[split]) != expected:
            raise RuntimeError("ImageFolder and image inventory disagree")
        identity[split] = {k: stats[k] for k in ("fingerprint", "fingerprint_kind", "images", "classes")}
        receipt = read_json(root / "_mpsae" / f"prepare_{'validation' if split == 'val' else split}.json")
        if receipt:
            identity[split]["revision"] = receipt.get("revision")
    if datasets["train"].class_to_idx != datasets["val"].class_to_idx:
        raise RuntimeError("ImageNet training/query label mappings disagree")
    identity["classes"] = canonical_hash(datasets["train"].class_to_idx)
    return datasets, identity


def transforms(spec, pretrained):
    from torchvision import transforms as T
    from ..benchmark.extract import evaluation_transform
    evaluation, metadata = evaluation_transform(spec, pretrained)
    training = T.Compose([
        T.RandomResizedCrop(spec.input_size, scale=(0.8, 1.0),
                            interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(metadata["mean"], metadata["std"])])
    return training, evaluation, {"evaluation": metadata,
        "training": "random_resized_crop_224_scale_0.8_1.0_ratio_3/4_4/3;hflip_0.5;normalize",
        "augmentation_rng": "seed+epoch*rows+sample_index"}


def extract_features(network, datasets, transform, spec, directory, *, identity,
                     batch_size, workers, device, precision):
    """FeatureSink resumes per split; no raw/decoded image duplication or NPZ copy."""
    import torch
    from ..embed.archive import FeatureSink
    from ..features import Bundle, Split
    from ..logs import Progress
    from .train import forward_features, autocast
    network.eval().requires_grad_(False)
    splits = {}
    manifests = {"backbone": spec.name}
    for split in ("train", "val"):
        metadata = {**identity, "split": split, "precision": precision, "batch_size": batch_size}
        with FeatureSink(directory, rows=len(datasets[split]), dim=spec.dim, identity=metadata,
                         stage=f"finetune/extract/{spec.name}/{split}") as sink:
            progress = Progress(len(datasets[split]), done=sink.cursor,
                                stage=f"extract/{spec.name}/{split}", what="images")
            if sink.remaining:
                batches = loader(datasets[split], transform, seed=0, epoch=0, batch_size=batch_size,
                                 workers=workers, skip=sink.cursor)
                # One inference precision context avoids per-batch setup.
                with torch.no_grad(), autocast(device, precision):
                    for images, labels in batches:
                        values = forward_features(network, images.to(device, non_blocking=True), spec)
                        sink.write(values.float().cpu().numpy(), labels.numpy())
                        progress.advance(len(labels))
            features, labels = sink.publish()
        name = "validation" if split == "val" else split
        manifests[name] = {"identity": canonical_hash(metadata), "rows": len(datasets[split])}
        splits[name] = Split(name, features, labels, manifests[name])
    return Bundle(spec.name, splits["train"], splits["validation"], manifests)


def close_bundle(bundle):
    for split in (bundle.train, bundle.validation):
        for array in (split.features, split.labels):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
