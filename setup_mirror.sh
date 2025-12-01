#!/bin/bash
# 配置Hugging Face镜像（中国用户）

export HF_ENDPOINT=https://hf-mirror.com
echo "✓ Hugging Face镜像已设置为: $HF_ENDPOINT"

# 如果需要永久设置，可以添加到 ~/.bashrc
# echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

