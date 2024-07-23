import os
from functools import partial

import common
import process
import transformers
from datasets import load_dataset
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
        WANDB_PROJECT="inno-gpt2-trainer",
        MODEL_ID="openai-community/gpt2-large",
        DSET_ID="HuggingFaceH4/no_robots",
    )
    cfg.update_env()
    return cfg


def get_model_via_config(cfg: common.Config):
    """
    get_model_via_config _summary_

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

    model = get_model_via_config(cfg=cfg)
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
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        fp16=True,
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
