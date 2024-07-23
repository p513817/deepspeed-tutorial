import json
import os
from pathlib import Path

from schema import SecretConfig


def load_secret_from_path(secret_config_path: str) -> SecretConfig:
    with open(secret_config_path, "r") as file:
        data = json.load(file)
        return SecretConfig(**data)


def get_secret_path_from_env(env: str = "INNO_DS_SECRET") -> str:
    secret_path = os.environ.get(env, None)

    if not secret_path:
        raise EnvironmentError(f"Can't not find environment key: {env}")

    if not Path(secret_path).exists():
        raise FileNotFoundError(
            f"Can't find configuration for secret token ({str(secret_path)})"
        )
    return secret_path


def get_secret(env: str = "INNO_DS_SECRET") -> SecretConfig:
    secret_path = get_secret_path_from_env(env=env)
    return load_secret_from_path(secret_path)
