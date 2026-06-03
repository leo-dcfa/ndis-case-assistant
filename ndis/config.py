"""Project configuration loader.

Loads config/required_fields.yaml and config/model.yaml at import time.
Returns frozen dataclasses so the config can't be mutated after loading.

Usage:
    from src.config import cfg

    # Required NDIS fields
    for field in cfg.required_fields:
        print(field.name, field.type)

    # Model training params
    lr = cfg.model.training.learning_rate
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Internal loader (private module-level state)
# ---------------------------------------------------------------------------

_CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"


@dataclass(frozen=True)
class FieldSpec:
    """A single required field definition from required_fields.yaml."""

    name: str
    type_name: str
    required: bool
    description: str
    format_hint: Optional[str] = None
    options: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RequiredFieldsConfig:
    """Parsed from config/required_fields.yaml."""

    fields: list[FieldSpec] = field(default_factory=list)
    structure_order: list[str] = field(default_factory=list)
    tone_guidelines: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class TrainingConfig:
    base_model_name: str = "Qwen3-8B"
    base_model_path: Optional[str] = None
    trust_remote_code: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    qlora_bitwidth: int = 4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    fp16: bool = True
    bf16: bool = False
    optim: str = "paged_adamw_8bit"


@dataclass(frozen=True)
class ServingConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 2048
    dtype: str = "float16"


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration for the project."""

    required_fields: RequiredFieldsConfig = field(default_factory=RequiredFieldsConfig)
    model: TrainingConfig = field(default_factory=TrainingConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_required_fields(path: pathlib.Path) -> RequiredFieldsConfig:
    """Load and parse config/required_fields.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"required_fields.yaml not found at {path}")

    cfg_dict = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(cfg_dict, dict), "required_fields.yaml must be a YAML mapping"

    fields = []
    for item in cfg_dict.get("required_fields", []):
        assert isinstance(item, dict), "Each field spec must be a mapping"
        fields.append(FieldSpec(
            name=item["field"],
            type_name=item["type"],
            required=item.get("required", False),
            description=item.get("description", ""),
            format_hint=item.get("format"),
            options=item.get("options", []),
        ))

    order = cfg_dict.get("structure_order", [f.name for f in fields if f.required])
    tone = cfg_dict.get("tone_guidelines", {})
    assert isinstance(tone, dict)

    return RequiredFieldsConfig(fields=fields, structure_order=order, tone_guidelines=tone)


def _load_model_config(path: pathlib.Path) -> TrainingConfig:
    """Load and parse config/model.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"model.yaml not found at {path}")

    cfg_dict = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(cfg_dict, dict), "model.yaml must be a YAML mapping"

    # Base model
    base = cfg_dict.get("base_model", {})
    base_model_name = base.get("name", "Qwen3-8B")
    base_model_path = base.get("path")  # None is valid (HF lookup)
    trust_remote_code = base.get("trust_remote_code", True)

    # LoRA
    lora_dict = cfg_dict.get("lora_config", {})
    lora = LoRAConfig(
        r=lora_dict.get("r", 16),
        alpha=lora_dict.get("alpha", 32),
        dropout=lora_dict.get("dropout", 0.05),
        target_modules=lora_dict.get("target_modules", []),
        bias=lora_dict.get("bias", "none"),
        task_type=lora_dict.get("task_type", "CAUSAL_LM"),
    )

    # Training
    train = cfg_dict.get("training", {})
    training = TrainingConfig(
        base_model_name=base_model_name,
        base_model_path=base_model_path,
        trust_remote_code=trust_remote_code,
        lora=lora,
        qlora_bitwidth=train.get("qlora_bitwidth", 4),
        batch_size=train.get("batch_size", 2),
        gradient_accumulation_steps=train.get("gradient_accumulation_steps", 8),
        learning_rate=float(train.get("learning_rate", 2.0e-4)),
        num_epochs=train.get("num_epochs", 3),
        warmup_ratio=train.get("warmup_ratio", 0.1),
        weight_decay=float(train.get("weight_decay", 0.01)),
        max_seq_length=int(train.get("max_seq_length", 2048)),
        fp16=train.get("fp16", True),
        bf16=train.get("bf16", False),
        optim=train.get("optim", "paged_adamw_8bit"),
    )

    return training


def _load_serving_config(cfg_dict: dict) -> ServingConfig:
    """Load serving config from model.yaml top-level 'serving' key."""
    serve = cfg_dict.get("serving", {})
    return ServingConfig(
        tensor_parallel_size=int(serve.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(serve.get("gpu_memory_utilization", 0.9)),
        max_model_len=int(serve.get("max_model_len", 2048)),
        dtype=serve.get("dtype", "float16"),
    )


def load_config(config_dir: Optional[pathlib.Path] = None) -> AppConfig:
    """Load all configuration files and return a frozen AppConfig.

    Args:
        config_dir: Path to the config directory. Defaults to config/ relative
            to this module's parent.

    Returns:
        A frozen AppConfig instance (dataclass — immutable).
    """
    if config_dir is None:
        config_dir = _CONFIG_DIR

    rf_path = config_dir / "required_fields.yaml"
    mf_path = config_dir / "model.yaml"

    req_cfg = _load_required_fields(rf_path)

    with open(mf_path, encoding="utf-8") as f:
        model_dict = yaml.safe_load(f)

    model_cfg = _load_model_config(mf_path)
    serving_cfg = _load_serving_config(model_dict if isinstance(model_dict, dict) else {})

    return AppConfig(
        required_fields=req_cfg,
        model=model_cfg,
        serving=serving_cfg,
    )


# ---------------------------------------------------------------------------
# Module-level accessor (read-only — config is loaded once on first import)
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    """Lazy access to cfg for convenience: from src.config import cfg."""
    if name == "cfg":
        return load_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_required_field_specs() -> list[FieldSpec]:
    """Return the list of required field specs (shorthand)."""
    return load_config().required_fields.fields


def get_structure_order() -> list[str]:
    """Return field names in the prescribed order."""
    return load_config().required_fields.structure_order
