import os
from os.path import exists, splitext
from pathlib import Path

from pydantic import BaseModel


def add_suffix_to_duplicates(filename: str | Path) -> str | Path:
    iteration = 1
    new_filename = filename
    while exists(new_filename):
        name, ext = splitext(new_filename)
        name_with_dash = name.rsplit("-", 1)
        if len(name_with_dash) == 2:
            org_name, org_iter = name_with_dash
            iteration = int(org_iter) + 1
            name = org_name
        new_filename = f"{name}-{iteration}{ext}"

    if isinstance(filename, str):
        return new_filename
    else:
        return Path(new_filename)


def get_run_name_via_env(cur_file: str = __file__, env: str = "RUN_NAME") -> str:
    run_name = os.environ.get(env, "")
    return run_name if run_name != "" else add_suffix_to_duplicates(Path(__file__).stem)


def get_env(env_key: str) -> str:
    return os.environ.get(env_key, "")


class Config(BaseModel):
    RUN_NAME: str
    WANDB_PROJECT: str | None = None
    MODEL_ID: str | None = None
    DSET_ID: str | None = None
    RESULT_DIR: str = "/opt/inno/results"

    NCCL_P2P_DISABLE: str = "1"
    NCCL_IB_DISABLE: str = "1"
    TOKENIZERS_PARALLELISM: str = "True"

    OUTPUT_DIR: str = ""
    WANDB_RUN_ID: str = ""
    WANDB_NOTES: str = ""

    def model_post_init(self, __context):
        if self.OUTPUT_DIR == "":
            self.OUTPUT_DIR = str(Path(self.RESULT_DIR) / self.RUN_NAME)
        if self.WANDB_RUN_ID == "":
            self.WANDB_RUN_ID = self.RUN_NAME

    def update_env(self):
        os.environ.update(self.model_dump())


if __name__ == "__main__":
    run_name = get_run_name_via_env(env="RUN_NAME")
    cfg = Config(
        RUN_NAME=run_name,
        WANDB_PROJECT="inno-llama3-8B",
        MODEL_ID="meta-llama/Meta-Llama-3-8B",
        DSET_ID="HuggingFaceH4/no_robots",
    )
    cfg.update_env()
    print(os.environ)
