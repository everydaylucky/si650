#!/bin/bash
# 解压 splits_data.zip 到 data/full 目录

cd /hy-tmp/final_test
mkdir -p data/full

# 方法1: 如果文件在远程服务器
# ssh -p 25545 root@i-2.gpushare.com 'cd /root/ && unzip -o splits_data.zip' | tar -xzf - -C data/full/

# 方法2: 如果文件已经在本地
if [ -f "splits_data.zip" ]; then
    echo "在本地找到 splits_data.zip，解压到 data/full..."
    unzip -o splits_data.zip -d data/full/
    echo "✓ 解压完成"
elif [ -f "/root/splits_data.zip" ]; then
    echo "在 /root/ 找到 splits_data.zip，解压到 data/full..."
    unzip -o /root/splits_data.zip -d data/full/
    echo "✓ 解压完成"
else
    echo "未找到 splits_data.zip，尝试从远程服务器下载并解压..."
    # 方法3: 先下载再解压
    ssh -o StrictHostKeyChecking=no -p 25545 root@i-2.gpushare.com 'cd /root/ && cat splits_data.zip' | unzip - -d data/full/
    echo "✓ 解压完成"
fi

# 检查解压结果
if [ -d "data/full" ]; then
    echo ""
    echo "解压后的文件:"
    ls -lh data/full/ | head -20
fi
