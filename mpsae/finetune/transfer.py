"""Frozen EuroSAT/SOP evaluation of the *jointly fine-tuned* backbone and SAE."""
from dataclasses import replace
import os

from ..config import BACKBONES
from ..io import atomic_json, canonical_hash, exclusive_lock, read_json, sha256_file


def budgets(config, cell, smoke):
    return (2, cell.k) if smoke else config.campaign.budgets("imagenet", "eval_k", cell.backbone)


def expected_points(config, smoke=False):
    return (2 * len(config.transfer["datasets"]) * sum(len(budgets(config, c, smoke)) for c in config.cells())
            if config.transfer["enabled"] else 0)


def smoke_spec(spec):
    return replace(spec, dim=16, recipe="mrl", k_list=(2, 8, 12, 16), train_k_list=(8, 12),
                   e2_anchors=(8, 12), preset=replace(spec.preset, hidden_dim=64, k_aux=8, ot_microbatch=4))


class _Rows:
    def __init__(self, source, transform, start=0):
        self.source, self.transform, self.start = source, transform, start

    def __len__(self):
        return self.source.rows - self.start

    def __getitem__(self, index):
        import torch
        from ..benchmark.data import open_rgb
        i = self.start + index
        image = self.source.dataset[i]["image"]
        return self.transform(image if isinstance(image, torch.Tensor) else open_rgb(image)), self.source.labels[i]


def load_target(name, root, config, smoke):
    from ..benchmark.data import DatasetBundle, DatasetSplit, DATASETS, load_dataset_bundle
    if not smoke:
        return load_dataset_bundle(name, root / "cache" / "transfer_data",
            revision=config.transfer["revisions"].get(name, "main"), token=os.environ.get("HF_TOKEN"))
    import torch
    generator = torch.Generator().manual_seed(73)
    spec, splits = DATASETS[name], {}
    for split in dict.fromkeys((spec.gallery_split, spec.query_split)):
        rows = 32 if split == spec.gallery_split else 16
        splits[split] = DatasetSplit(split,
            [{"image": torch.randn(3, 8, 8, generator=generator)} for _ in range(rows)],
            [f"{name}:{split}:{i}" for i in range(rows)], [i % 4 for i in range(rows)])
    identity = {"recipe": "synthetic-joint-transfer-v1", "dataset": name}
    return DatasetBundle(name, spec.protocol, splits, spec.gallery_split, spec.query_split,
                         spec.exclude_self, identity, identity)


def extract(network, spec, data, transform, root, *, checkpoint_id, preprocessing,
            precision, device, batch_size, workers):
    import torch
    from torch.utils.data import DataLoader
    from ..benchmark.extract import FeatureBundle, FeatureSplit
    from ..embed.archive import FeatureSink, library_versions
    from ..logs import Progress
    from .train import autocast, forward_features
    network.eval().requires_grad_(False)
    identity = {"recipe": "joint-transfer-v1", "checkpoint_sha256": checkpoint_id,
        "dataset": data.identity, "backbone": spec.name, "preprocessing": preprocessing,
        "precision": precision, "device_type": torch.device(device).type,
        "batch_size": batch_size, "libraries": library_versions()}
    loaded = {}
    for name in dict.fromkeys((data.gallery_split, data.query_split)):
        source = data.splits[name]
        split_identity = {**identity, "split": name, "ids": canonical_hash(source.ids),
                          "labels": canonical_hash(source.labels)}
        with FeatureSink(root / "cache" / "transfer_features", rows=source.rows, dim=spec.dim,
                         identity=split_identity, stage=f"transfer/{data.name}/{spec.name}/{name}") as sink:
            if sink.remaining:
                loader = DataLoader(_Rows(source, transform, sink.cursor), batch_size=batch_size,
                    num_workers=workers, shuffle=False, pin_memory=torch.device(device).type == "cuda")
                progress = Progress(source.rows, done=sink.cursor, stage=f"transfer/{data.name}", what="images")
                with torch.no_grad(), autocast(device, precision):
                    for images, labels in loader:
                        values = forward_features(network, images.to(device, non_blocking=True), spec)
                        sink.write(values.float().cpu().numpy(), labels.numpy())
                        progress.advance(len(images))
            features, labels = sink.publish()
        loaded[name] = FeatureSplit(name, features, labels, source.ids, split_identity)
    return FeatureBundle(data.name, data.protocol, spec.name, loaded[data.gallery_split],
                         loaded[data.query_split], data.exclude_self, identity,
                         {"dataset": data.manifest, "extraction": identity})


def close_features(features):
    seen = set()
    for split in (features.gallery, features.query):
        for array in (split.features, split.labels):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and id(mapping) not in seen:
                seen.add(id(mapping))
                mapping.close()


def evaluate_cell(config, root, cell, spec, network, sae, transform, preprocessing, summary, *,
                  device, workers, extract_batch, data_cache, smoke=False, on_result=None):
    import torch
    from ..benchmark.run import evaluate_point
    checkpoint_id = sha256_file(root / "runs" / cell.slug / "checkpoint.pt")
    checkpoint = {"checkpoint_id": checkpoint_id, "signature": summary["signature"]}
    records = []
    sae.eval().requires_grad_(False).to(device)
    for name in config.transfer["datasets"]:
        if name not in data_cache:
            data_cache[name] = load_target(name, root, config, smoke)
        data = data_cache[name]
        network.to(device)
        features = extract(network, spec, data, transform, root, checkpoint_id=checkpoint_id,
            preprocessing=preprocessing, precision="float32" if smoke else config.training["precision"],
            device=device, workers=workers, batch_size=extract_batch)
        try:
            network.to("cpu")
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
            for k in budgets(config, cell, smoke):
                for task in ("knn_classification", "retrieval"):
                    result = evaluate_point(features, spec, root, cell=cell, checkpoint=checkpoint,
                        model=sae, k=k, device=device, task=task, knn=config.transfer["knn"], smoke=smoke)
                    # Backbones differ across methods by design after joint training.
                    # Group comparisons by the common dataset/preprocessing/protocol,
                    # retaining exact checkpoint IDs in each evaluation's identity.
                    comparison = {"dataset": data.identity, "preprocessing": preprocessing,
                        "precision": features.identity["precision"], "device_type": features.identity["device_type"],
                        "libraries": features.identity["libraries"], "regime": "joint_finetuned", "task": task,
                        "classifier": result.get("classifier"), "protocol": data.protocol}
                    result = {**result, "mode": "joint_finetuned", "comparison_id": canonical_hash(comparison),
                        "representation_training": "ImageNet_joint_finetune", "training_on_target": False}
                    records.append(result)
                    if on_result:
                        on_result(result)
        finally:
            close_features(features)
    return records


def report(config, root, records, *, smoke=False, failures=()):
    from ..benchmark.report import build_report
    output = root / "reports" / config.plan()["plan_id"] / "transfer"
    result = build_report(records, output, expected_points=expected_points(config, smoke),
                          expected_seeds=config.campaign.seeds, failures=failures)
    try:
        result["figures"].update(plot_overview(config, records, output / "figures", smoke=smoke))
    except Exception as error:
        result["plot_failures"].append(f"transfer overview: {type(error).__name__}: {error}")
        result["complete"] = False
    result.update(scientific=not smoke, regime="ImageNet_joint_finetune_then_frozen_transfer")
    atomic_json(output / "manifest.json", result)
    (output / "README.txt").write_text(
        ("SYNTHETIC SMOKE TEST; not scientific results.\n" if smoke else "")
        + "EuroSAT: train support, test queries. SOP: test gallery and queries, self image excluded.\n"
          "SOP weighted kNN is leave-one-out product voting, separate from standard retrieval.\n"
        + f"Neighbour counts {config.transfer['knn']['n_neighbors']} differ from inference sparsity K. "
          "No target parameter updates.\n"
          "Each curve holds the fine-tuned checkpoint fixed and varies inference K.\n", encoding="utf-8")
    return result


def plot_overview(config, records, output, *, smoke=False):
    """Put weighted accuracy and retrieval side by side, with separate datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..benchmark.report import aggregate_records, _curve
    from ..report.figures import _dedup_legend
    from ..report.style import apply_style, backbone_label, save_figure, seed_caption
    apply_style()
    rows = aggregate_records(records, expected_seeds=config.campaign.seeds)
    neighbours = max(config.transfer["knn"]["n_neighbors"])
    metrics = [("knn_classification", f"knn_{neighbours}_top1", f"w{neighbours}-NN (%)"),
               ("retrieval", "recall_at_1", "R@1 (%)"), ("retrieval", "map_at_r", "mAP@R (%)")]
    paths = {}
    for backbone, train_k in sorted({(r["backbone"], r["train_k"]) for r in rows}):
        subset = [r for r in rows if (r["backbone"], r["train_k"]) == (backbone, train_k)]
        names = [n for n in config.transfer["datasets"] if any(r["dataset"] == n for r in subset)]
        fig, axes = plt.subplots(len(names), 3, figsize=(10, 2.5 * len(names) + 0.5),
                                 squeeze=False, layout="constrained")
        try:
            for i, dataset in enumerate(names):
                for j, (task, metric, label) in enumerate(metrics):
                    ax = axes[i, j]
                    selected = [r for r in subset if r["dataset"] == dataset and r["task"] == task]
                    for method in config.campaign.methods:
                        _curve(ax, [r for r in selected if r["method"] == method], method, metric)
                    ticks = sorted({r["eval_k"] for r in selected})
                    ax.set_xscale("log", base=2)
                    ax.set_xticks(ticks, [str(k) for k in ticks])
                    ax.set_xlabel("Inference $K$")
                    ax.set_ylabel(label)
                    ax.set_title("EuroSAT" if dataset == "eurosat" else
                                 "SOP (LOO)" if task == "knn_classification" else "SOP")
            title = f"{backbone_label(backbone)} · Train $K$={train_k}"
            fig.suptitle(("Synthetic test · " if smoke else "") + title)
            _dedup_legend(fig, list(axes.flat), ncol=3)
            name = f"transfer_{backbone}_train-k{train_k}"
            caption = (f"{title}. Joint ImageNet fine-tuning followed by frozen sparse transfer. "
                "No target parameter updates. EuroSAT uses train support and test queries; "
                "same-class images are relevant for retrieval. SOP uses test products with "
                "each query excluded; LOO weighted voting is separate from standard retrieval. "
                f"DINO weighted {neighbours}-NN, T={config.transfer['knn']['temperature']}. "
                "Curves vary inference K of a fixed checkpoint. " + seed_caption(config.campaign.seeds))
            paths[name] = [str(p) for p in save_figure(fig, output, name, caption=caption)]
        finally:
            plt.close(fig)
    return paths


def source_record(config, root, cell, smoke=False):
    from .config import VERSION
    directory = root / "runs" / cell.slug
    summary, saved_config = read_json(directory / "summary.json"), read_json(directory / "config.json")
    path = directory / "checkpoint.pt"
    if not summary or not summary.get("complete") or not saved_config or not path.is_file():
        raise FileNotFoundError(f"missing completed joint fine-tuning checkpoint: {directory}")
    identity = summary.get("identity")
    if (canonical_hash(identity) != summary.get("signature") or saved_config.get("identity") != identity
            or saved_config.get("signature") != summary["signature"] or identity.get("format") != VERSION
            or identity.get("cell") != cell.__dict__ or identity.get("smoke") != smoke
            or summary.get("steps") != identity["schedule"]["total_steps"]):
        raise RuntimeError(f"{cell.slug}: inconsistent joint checkpoint metadata")
    options = config.training_options(cell)
    if smoke:
        options.update(precision="float32", batch_size=4, epochs=1, checkpoint_every=1)
    if identity["training"] != options:
        raise RuntimeError(f"{cell.slug}: checkpoint training settings do not match the campaign")
    return summary, path


def run_transfer(config, workspace, *, device="cuda:0", workers=8, extract_batch=64, smoke=False):
    """Evaluate saved joint models without ImageNet images or source SAE files."""
    import torch
    from torch import nn
    from ..benchmark.extract import _network
    from ..model import build_sae
    from ..logs import get_logger, setup_logging
    from .data import transforms
    from .run import output_root
    if not config.transfer["enabled"]:
        raise ValueError("transfer.enabled must be true")
    if workers < 0 or extract_batch < 1:
        raise ValueError("workers must be nonnegative and extract_batch positive")
    root = output_root(config, workspace, smoke).resolve()
    setup_logging(root / "logs", name="transfer")
    with exclusive_lock(root / "campaign.lock", timeout=0):
        sources = {c.slug: source_record(config, root, c, smoke) for c in config.cells()}
        records, datasets, failures = [], {}, []
        path = root / "campaigns" / config.name / config.plan()["plan_id"] / "transfer.json"
        def persist(record):
            records.append(record)
            atomic_json(path, {"records": records, "scientific": not smoke})
        try:
            for cell in config.cells():
                get_logger("transfer").info("Evaluating joint checkpoint %s", cell.slug)
                summary, checkpoint_path = sources[cell.slug]
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if (canonical_hash(saved.get("identity")) != summary["signature"]
                        or saved.get("format") != summary["identity"]["format"]
                        or saved.get("signature") != summary["signature"]
                        or saved.get("next_step") != summary["steps"]):
                    raise RuntimeError(f"{cell.slug}: checkpoint tensors and summary disagree")
                spec = smoke_spec(BACKBONES[cell.backbone]) if smoke else BACKBONES[cell.backbone]
                if smoke:
                    network = nn.Sequential(nn.Flatten(), nn.Linear(192, 16), nn.Tanh())
                    transform, preprocessing = nn.Identity(), {"synthetic": True}
                else:
                    network, pretrained, _ = _network(spec, root / "cache" / "weights", torch.device("cpu"),
                        os.environ.get("HF_TOKEN"), summary["identity"]["source"]["backbone_weights"])
                    _, transform, preprocessing = transforms(spec, pretrained)
                    if preprocessing != summary["identity"]["preprocessing"]:
                        raise RuntimeError(f"{cell.slug}: backbone preprocessing changed")
                network.load_state_dict(saved["backbone"], strict=True)
                sae = build_sae(spec.dim, spec.preset.hidden_dim, seed=cell.seed)
                sae.load_state_dict(saved["sae"], strict=True)
                del saved
                network.to(device)
                evaluate_cell(config, root, cell, spec, network, sae, transform, preprocessing, summary,
                    device=device, workers=workers, extract_batch=extract_batch, data_cache=datasets,
                    smoke=smoke, on_result=persist)
                del network, sae
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
        except BaseException as error:
            failures.append(f"{type(error).__name__}: {error}")
            raise
        finally:
            atomic_json(path, {"records": records, "scientific": not smoke})
            result = report(config, root, records, smoke=smoke, failures=failures)
        return result
