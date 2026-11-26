"""Names and provenance for the supported research models (no heavy imports)."""

MINI_ARCHITECTURES = {
    "deepseek_v41_mini": ("deepseek", "DeepSeekMiniEncoder"),
    "glm54_mini": ("glm", "GLMMiniEncoder"),
    "qwen4_mini": ("qwen", "QwenMiniEncoder"),
    "timesfm3_mini": ("timesfm", "TimesFMMiniEncoder"),
}
NEURAL_MODELS = ("transformer", *MINI_ARCHITECTURES)
TRAINABLE_MODELS = ("lightgbm", *NEURAL_MODELS)
MODEL_NAMES = (*TRAINABLE_MODELS, "timesfm3_pretrained")


def build_mini_encoder(name, **kwargs):
    from importlib import import_module

    if name not in MINI_ARCHITECTURES:
        raise ValueError(f"Unknown mini architecture: {name}")
    module, cls = MINI_ARCHITECTURES[name]
    return getattr(import_module(f"financial_ts.architectures.{module}"), cls)(**kwargs)
