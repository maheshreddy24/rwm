"""Size presets for RecurrentWorldModel / RecurrentWorldModelCLS.

`CORE_VARIANTS` (used by `RecurrentWorldModel` / trainer_tf) scales only the
recurrent core. The vision encoder is fixed -- pass `encoder_name`/`encoder`
directly if you want a different one -- and the decoder keeps its own
defaults independently of `variant` too. A trainable CNN adapter sits
between the (possibly frozen) encoder and the core, so the core's width
(`core_dim`) no longer has to match whatever encoder is loaded -- it's
derived from `preserve_ratio` instead (see `RecurrentWorldModel`), so
variants only scale depth/head-count here, not width.

`MODEL_VARIANTS` (used by `RecurrentWorldModelCLS` / trainer_cls) is the
older bundle that also swaps the encoder and decoder sizes per variant.

`resolve_variant` lets a config say `variant: base` and still override any
individual field -- explicit config values win over the preset.
"""

CORE_VARIANTS = {
    "s": dict(core_layers=4, core_heads=8, core_mlp=None),
    "base": dict(core_layers=6, core_heads=12, core_mlp=None),
    "l": dict(core_layers=8, core_heads=16, core_mlp=None),
}

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


def resolve_variant(model_config: dict, variants: dict = MODEL_VARIANTS) -> dict:
    """Pop `variant` out of a model config dict and layer it under the rest.

    `{**variants[variant], **model_config}` so any field the config sets
    explicitly overrides the preset, and fields it leaves out fall back to
    the variant's defaults.
    """
    model_config = dict(model_config)
    variant = model_config.pop("variant", None)
    if variant is None:
        return model_config
    if variant not in variants:
        raise ValueError(f"unknown model variant {variant!r}; choose from {list(variants)}")
    return {**variants[variant], **model_config}
