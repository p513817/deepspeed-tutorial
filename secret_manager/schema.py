from pydantic import BaseModel


class HuggingfaceConfig(BaseModel):
    token: str


class WandbConfig(BaseModel):
    token: str


class SecretConfig(BaseModel):
    container: str
    huggingface: HuggingfaceConfig
    wandb: WandbConfig
