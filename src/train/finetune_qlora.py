"""QLoRA fine-tune (Phase 3) — TRL SFTTrainer + peft, 4-bit, single GPU.

Reads LoRA / training hyperparameters from ``config/model.yaml`` and trains on the
chat-formatted ``train.jsonl`` split, saving a LoRA adapter. The base model is
``config/model.yaml``'s ``base_model.name`` unless overridden with --base-model
(handy for validating the pipeline on a small local model before the 8B run).

    # quick pipeline validation on a local model
    uv run python -m train.finetune_qlora --base-model Qwen/Qwen2.5-3B-Instruct \
        --max-steps 30 --out runs/adapters/smoke

    # real run
    uv run python -m train.finetune_qlora --base-model Qwen/Qwen3-8B \
        --out runs/adapters/qwen3-8b-v1

Heavy imports are done inside main() so the module stays cheap to import.
"""

from __future__ import annotations

import argparse
import pathlib

from ndis.config_loader import get_config


def main(argv: list[str] | None = None) -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from train.format_dataset import load_chat_dataset

    cfg = get_config().model
    parser = argparse.ArgumentParser(description="QLoRA fine-tune for NDIS case notes.")
    parser.add_argument("--base-model", default=cfg.base_model_path or cfg.base_model_name)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--out", default="runs/adapters/v1")
    parser.add_argument("--epochs", type=float, default=cfg.num_epochs)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs (smoke runs).")
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--grad-accum", type=int, default=cfg.gradient_accumulation_steps)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument("--max-seq-len", type=int, default=cfg.max_seq_length)
    args = parser.parse_args(argv)

    splits_dir = pathlib.Path(args.splits_dir)
    train_ds: Dataset = load_chat_dataset(splits_dir, "train")
    try:
        eval_ds: Dataset | None = load_chat_dataset(splits_dir, "val")
    except FileNotFoundError:
        eval_ds = None

    print(
        f"[train] base={args.base_model}  train={len(train_ds)}  "
        f"val={len(eval_ds) if eval_ds else 0}  out={args.out}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )

    sft_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_length=args.max_seq_len,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        optim=cfg.optim,
        report_to="none",
    )
    if eval_ds is not None:
        sft_kwargs["eval_strategy"] = "epoch"
    # Train only on the assistant turn when the TRL version supports it.
    try:
        sft_config = SFTConfig(assistant_only_loss=True, **sft_kwargs)
    except TypeError:
        sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[train] saved LoRA adapter -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
