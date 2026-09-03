"""Size presets for RecurrentWorldModel / RecurrentWorldModelCLS.

Each preset fixes the vision encoder backbone plus the core (RNN) and
decoder head dims/depths sized to go with it. `resolve_variant` lets a
config say `variant: base` and still override any individual field --
explicit config values win over the preset.
"""

MODEL_VARIANTS = {
    "s": dict(
        encoder_name="facebook/dinov2-with-registers-small",
        core_layers=4, core_heads=8, core_mlp=None,
        dec_dim=512, dec_layers=3, dec_heads=8, dec_mlp=2048,
    ),
    "base": dict(
        encoder_name="facebook/dinov2-with-registers-base",
        core_layers=6, core_heads=12, core_mlp=None,
        dec_dim=768, dec_layers=4, dec_heads=12, dec_mlp=3072,
    ),
    "l": dict(
        encoder_name="facebook/dinov2-with-registers-large",
        core_layers=8, core_heads=16, core_mlp=None,
        dec_dim=1024, dec_layers=6, dec_heads=16, dec_mlp=4096,
    ),
}


def resolve_variant(model_config: dict) -> dict:
    """Pop `variant` out of a model config dict and layer it under the rest.

    `{**MODEL_VARIANTS[variant], **model_config}` so any field the config
    sets explicitly overrides the preset, and fields it leaves out fall
    back to the variant's defaults.
    """
    model_config = dict(model_config)
    variant = model_config.pop("variant", None)
    if variant is None:
        return model_config
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown model variant {variant!r}; choose from {list(MODEL_VARIANTS)}")
    return {**MODEL_VARIANTS[variant], **model_config}
