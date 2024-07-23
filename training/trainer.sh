#!/bin/bash

# Function to find json files in the configs folder and return their real paths
function get_configs(){
    find "$(realpath $CFG_DIR)" -name "*.json"
}

# Function to parse config name and extension
function parser_config_name(){
    local filename=$(basename -- "$1")
    local name="${filename%.*}"
    echo "$name"
}

# Deep Speed Python Script
echo ${DS_EXEC}
if [ -z "${DS_EXEC}" ]; then
  echo "Please set up executable file. e.g. DS_EXEC={your deepspeed python script}"
  exit 1
fi

LOG_DIR=${LOG_DIR:-"logs"}
CFG_DIR=${CFG_DIR:-"configs"}

# Loop through each json config and run the specified command
for config_path in $(get_configs); do
    config_name=$(parser_config_name "$config_path")
    if [ "$DEBUG" == true ]; then
        echo "RUN_NAME=${config_name} DS_CONFIG=${config_path} ds --num_gpus 2 ${DS_EXEC}"
    else
        if [ ! -d "$LOG_DIR" ]; then
            mkdir -p $LOG_DIR
        fi
        RUN_NAME=${config_name} DS_CONFIG=${config_path} ds --num_gpus 2 ${DS_EXEC} > ${LOG_DIR}/${config_name}.log
    fi
done
