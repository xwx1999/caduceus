#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Activate conda env
# shellcheck source=${HOME}/.bashrc disable=SC1091

# 脚本使用 source 命令来激活 Conda 环境。CONDA_SHELL 是一个环境变量，通常由 Conda 安装过程设置，指向 Conda 的 shell 脚本。
# 接下来的 if 语句检查 CONDA_PREFIX 环境变量是否存在，以及它是否包含字符串 "/caduceus_env"。这个字符串应该对应于激活的 Conda 环境名称。
# 如果 CONDA_PREFIX 为空或者不包含 "/caduceus_env"，则脚本会尝试激活名为 caduceus_env 的 Conda 环境。如果该环境已经被激活但路径不匹配，脚本会先停用当前环境，然后重新激活正确的环境。

source "${CONDA_SHELL}"
if [ -z "${CONDA_PREFIX}" ]; then
    conda activate caduceus_env
 elif [[ "${CONDA_PREFIX}" != *"/caduceus_env" ]]; then
  conda deactivate
  conda activate caduceus_env
fi

# Add root directory to PYTHONPATH to enable module imports
# 最后，脚本将当前工作目录（${PWD}，即 print working directory 的缩写）添加到 PYTHONPATH 环境变量中。这样做可以让 Python 导入当前目录下的模块，这对于运行脚本或模块测试非常有用。
export PYTHONPATH="${PWD}"
