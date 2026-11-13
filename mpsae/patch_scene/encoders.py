"""Vision towers only. No pooled image embeddings or image-text projection."""
import torch

from .config import ENCODERS


def load_encoder(name, revision, device):
    from transformers import CLIPVisionModel, Dinov2Model, SiglipVisionModel, AutoImageProcessor
    repo, _ = ENCODERS[name]
    cls = {'clip': CLIPVisionModel, 'dinov2': Dinov2Model, 'siglip2': SiglipVisionModel}[name]
    # Load in fp32, use autocast for the frozen forward. Never create the text tower.
    model = cls.from_pretrained(repo, revision=revision, attn_implementation='sdpa')
    model = model.eval().requires_grad_(False).to(device)
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision)
    return model, (processor.image_mean, processor.image_std)


@torch.inference_mode()
def patch_tokens(model, images, name):
    options = {'interpolate_pos_encoding': True} if name != 'dinov2' else {}
    output = model(pixel_values=images, **options).last_hidden_state
    if name == 'clip':
        # HF CLIP applies post_layernorm only to pooled CLS by default.
        output = model.vision_model.post_layernorm(output[:, 1:])
    elif name == 'dinov2':
        output = output[:, 1:]  # chosen checkpoint has no register tokens
    # SigLIP has no CLS; last_hidden_state is already post-layernorm.
    patch = ENCODERS[name][1]
    expected = (images.shape[-2]//patch)*(images.shape[-1]//patch)
    if output.shape != (len(images), expected, 768) or not torch.isfinite(output).all():
        raise ValueError(f'invalid {name} patch tensor: {tuple(output.shape)}')
    return output.float()
