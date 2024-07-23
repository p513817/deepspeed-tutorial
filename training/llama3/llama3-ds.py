import os
from functools import partial

import common
import peft
import process
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def init_config(run_name: str | None = None) -> common.Config:
    if run_name is None:
        run_name = common.get_run_name_via_env(env="RUN_NAME")
    cfg = common.Config(
        RUN_NAME=run_name,
        WANDB_PROJECT="inno-llama3-8B",
        MODEL_ID="meta-llama/Meta-Llama-3-8B",
        DSET_ID="HuggingFaceH4/no_robots",
    )
    cfg.update_env()
    return cfg


def get_peft_model_via_config(cfg: common.Config) -> peft.PeftModelForCausalLM:
    """
    get_peft_model_via_config _summary_

    Args:
        cfg (common.Config): _description_

    Returns:
        peft.PeftModelForCausalLM: _description_

    Introduction:
        * lora_attn_modules: Specifies which components of the attention mechanism LoRA is applied to. Common choices include 'q_proj' (query projection) and 'v_proj' (value projection), as these are key areas where adaptations can significantly impact model performance by modifying how inputs are weighted and combined.
        * lora_rank: Determines the rank of the low-rank matrices that are used to approximate the original weight matrices in the specified modules. A lower rank means fewer parameters to train, which can speed up training and reduce overfitting but may limit the model’s capacity to learn complex patterns.
        * lora_alpha: A scaling factor that adjusts the magnitude of the updates applied through the LoRA parameters. This can be critical for balancing the influence of the LoRA-enhanced components on the model’s behavior.

    """
    # Get model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_ID,
        # torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Wrap to peft model
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.0,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    return model


def get_tokenizer(
    cfg: common.Config,
) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID, use_deepspeed=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    cfg = init_config()
    ds_config = common.get_env("DS_CONFIG")
    if not ds_config:
        raise ValueError(f"Not define deepspeed config: {ds_config}")
    if not os.path.exists(ds_config):
        raise FileNotFoundError(f"Can not find deepspeed config: {ds_config}")

    model = get_peft_model_via_config(cfg=cfg)
    tokenizer = get_tokenizer(cfg=cfg)
    ds = load_dataset(cfg.DSET_ID, trust_remote_code=True)
    tokenized_dataset = ds.map(
        partial(process.preprocess_with_tokenizer, tokenizer),
        batched=True,
        remove_columns=["prompt", "prompt_id", "messages", "category"],
    )
    training_arguments = TrainingArguments(
        run_name=cfg.RUN_NAME,
        output_dir=cfg.OUTPUT_DIR,
        eval_strategy="no",
        optim="adamw_torch",
        learning_rate=3e-4,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        fp16=True,
        max_grad_norm=0.3,
        warmup_steps=5,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="wandb",
        deepspeed=ds_config,
    )
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        args=training_arguments,
    )
    trainer.train()


if __name__ == "__main__":
    main()
