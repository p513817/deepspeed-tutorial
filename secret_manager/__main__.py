import utils
from huggingface_hub import login as hf_login
from wandb import login as wdb_login

if __name__ == "__main__":
    # Custom module for login
    serect = utils.get_secret(env="INNO_DS_SECRET")
    hf_login(serect.huggingface.token)
    wdb_login(key=serect.wandb.token, relogin=True, force=True)
