# deepspeed-tutorial
Deep Speed Tutorial For Newbie

# Requirements
1. [Docker Engine](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
    * [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/)
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

# Setup
1. Prepare the token configuration
    1. Copy `config/secret.json` -> `config/{user}.json`
    2. Modify token in `config/{user}.json`

2. Build Docker Image and Run Container with custom token
    ```bash
    ./docker/build.sh
    SECRET=./config/{user}.json ./docker/run.sh
    ```
    * Customize container name
        * `CNTR=testtest ./docker/run.sh` 
3. Login
    ```bash
    python3 secret_manager
    ```
    The cache file will store in `cache/huggingface` and `cache/wandb`

# Usage
1. Define deepspeed config
2. Run trainer to train several configs with deep speed
    ```bash
    cd training
    DS_EXEC="gpt2/gpt2-large-ds.py" \
    CFG_DIR="gpt2/configs" \
    ./trainer.sh
    ```