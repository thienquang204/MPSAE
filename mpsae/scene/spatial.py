"""Spatial pre-logit adapters for the exact registered pretrained backbones."""
from .core import resnet_spatial

from .config import STRIDES
RECIPES = {'ff2048': 'layer4_pre_avgpool',
           'convnext_tiny': 'forward_features_then_pointwise_prelogit_head_v1',
           'vit_b16': 'normalized_patch_tokens_excluding_prefix_v1',
           'dinov2_vits14': 'normalized_patch_tokens_excluding_prefix_v1'}


def loading_spec(config):
    from dataclasses import replace
    from ..config import BACKBONES
    spec = BACKBONES[config.backbone]
    # Only transformers accept img_size and require positional interpolation.
    # Convolutional models already accept arbitrary spatial input sizes.
    if config.backbone in ('vit_b16', 'dinov2_vits14'):
        return replace(spec, input_size=config.image_size)
    return spec


def normalization(spec):
    if spec.normalize is not None:
        return spec.normalize
    import timm
    cfg = timm.get_pretrained_cfg(spec.source.removeprefix('timm/'))
    return tuple(cfg.mean), tuple(cfg.std)


def spatial_features(network, images, backbone):
    if backbone == 'ff2048':
        return resnet_spatial(network, images)
    x = network.forward_features(images)
    if backbone == 'convnext_tiny':
        b, c, h, w = x.shape
        # Run the existing head on each one-cell map, preserving its LayerNorm/MLP.
        cells = x.permute(0, 2, 3, 1).reshape(-1, c, 1, 1)
        return network.forward_head(cells, pre_logits=True).reshape(b, h, w, -1)
    if backbone in ('vit_b16', 'dinov2_vits14'):
        if network.global_pool not in ('token', 'avg'):
            raise ValueError('unsupported transformer pooling rule')
        h, w = images.shape[-2]//STRIDES[backbone], images.shape[-1]//STRIDES[backbone]
        patches = x[:, network.num_prefix_tokens:]
        if patches.shape[1] != h*w:
            raise ValueError('patch-token count disagrees with spatial grid')
        patches = network.head_drop(network.fc_norm(patches))
        return patches.reshape(len(images), h, w, -1)
    raise ValueError(f'unsupported spatial backbone: {backbone}')
