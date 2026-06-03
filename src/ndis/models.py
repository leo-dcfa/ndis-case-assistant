"""NDIS data models: CaseNote schema + config loading."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


DEFAULT_CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type_name: str
    required: bool
    description: str
    format_hint: Optional[str] = None
    options: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RequiredFieldsConfig:
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
    required_fields: RequiredFieldsConfig = field(default_factory=RequiredFieldsConfig)
    model: TrainingConfig = field(default_factory=TrainingConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)


def load_config(config_dir: Optional[pathlib.Path] = None) -> AppConfig:
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR

    rf_path = config_dir / "required_fields.yaml"
    mf_path = config_dir / "model.yaml"

    req_cfg = _load_required_fields(rf_path)

    model_dict = yaml.safe_load(mf_path.read_text(encoding="utf-8"))
    if not isinstance(model_dict, dict):
        raise ValueError("model.yaml must contain a YAML mapping")

    base = model_dict.get("base_model", {})
    train = model_dict.get("training", {})
    serve = model_dict.get("serving", {})
    lora_d = model_dict.get("lora_config", {})

    lora = LoRAConfig(
        r=lora_d.get("r", 16),
        alpha=lora_d.get("alpha", 32),
        dropout=lora_d.get("dropout", 0.05),
        target_modules=lora_d.get("target_modules", []),
        bias=lora_d.get("bias", "none"),
        task_type=lora_d.get("task_type", "CAUSAL_LM"),
    )

    model = TrainingConfig(
        base_model_name=base.get("name", "Qwen3-8B"),
        base_model_path=base.get("path"),
        trust_remote_code=base.get("trust_remote_code", True),
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

    serving = ServingConfig(
        tensor_parallel_size=int(serve.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(serve.get("gpu_memory_utilization", 0.9)),
        max_model_len=int(serve.get("max_model_len", 2048)),
        dtype=serve.get("dtype", "float16"),
    )

    return AppConfig(required_fields=req_cfg, model=model, serving=serving)


def _load_required_fields(path: pathlib.Path) -> RequiredFieldsConfig:
    if not path.exists():
        raise FileNotFoundError(f"required_fields.yaml not found at {path}")

    cfg_dict = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(cfg_dict, dict), "required_fields.yaml must be a YAML mapping"

    fields = []
    for item in cfg_dict.get("required_fields", []):
        assert isinstance(item, dict)
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


class CaseNote(BaseModel):
    """Structured NDIS case note."""

    model_config = {"extra": "forbid"}

    participant_id: str = Field(description="De-identified participant reference")
    date_of_service: date = Field(description="Date the service was delivered")
    duration_minutes: int = Field(description="Duration of the session in minutes")
    service_type: str = Field(description="NDIS support category/title")
    goal_linkage: str = Field(description="Which participant goal this service supports")
    location: str = Field(description="Where the service was delivered")
    staff_presented_by: str = Field(description="Support worker name (de-identified)")
    participant_present: bool = Field(description="Whether the participant was present")
    narrative_summary: str = Field(description="Narrative of what occurred during the session")
    billable_evidence: str = Field(description="Evidence supporting billing")
    outcomes_achieved: List[str] = Field(description="Specific outcomes or progress noted")
    risk_management: Optional[str] = Field(default=None, description="Risks identified or interventions")
    follow_up_needed: bool = Field(description="Whether follow-up is indicated")
    follow_up_notes: Optional[str] = Field(default=None, description="Notes for next session")

    @property
    def structure_order(self) -> list[str]:
        cfg = load_config()
        return cfg.required_fields.structure_order

    @field_validator("service_type")
    @classmethod
    def _check_service_type(cls, v: str) -> str:
        cfg = load_config()
        svc_field = next((f for f in cfg.required_fields.fields if f.name == "service_type"), None)
        if svc_field and svc_field.options and v not in svc_field.options:
            raise ValueError(f"service_type must be one of {svc_field.options}, got '{v}'")
        return v

    @field_validator("date_of_service")
    @classmethod
    def _check_date_format(cls, v: date) -> date:
        if not isinstance(v, date):
            raise ValueError("date_of_service must be a date")
        return v

    @field_validator("duration_minutes")
    @classmethod
    def _check_duration(cls, v: int) -> int:
        if v <= 0 or v > 1440:
            raise ValueError("duration_minutes must be between 1 and 1439")
        return v

    @model_validator(mode="after")
    def check_required_present(self) -> "CaseNote":
        order = load_config().required_fields.structure_order
        for field_name in order:
            val = getattr(self, field_name)
            if isinstance(val, str) and not val.strip():
                raise ValueError(f"required field '{field_name}' is empty")
            if isinstance(val, list) and len(val) == 0:
                raise ValueError(f"required field '{field_name}' is empty list")
        return self

    def to_dict(self, exclude_none: bool = True) -> dict[str, Any]:
        return self.model_dump(exclude_none=exclude_none)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseNote":
        if isinstance(data.get("date_of_service"), str):
            data = dict(data)
            data["date_of_service"] = datetime.strptime(
                data["date_of_service"], "%Y-%m-%d"
            ).date()
        return cls(**data)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "CaseNote":
        return cls.from_dict(json.loads(line))
