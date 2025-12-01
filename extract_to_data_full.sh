#!/bin/bash
# 解压 splits_data.zip 到 data/full

cd /hy-tmp/final_test
mkdir -p data/full

echo "正在从远程服务器解压文件到 data/full..."
ssh -p 25545 root@i-2.gpushare.com 'cd /root/ && unzip -o splits_data.zip -d /hy-tmp/final_test/data/full/'

if [ $? -eq 0 ]; then
    echo "✓ 解压成功！"
    echo ""
    echo "解压后的文件："
    ls -lh data/full/ | head -20
else
    echo "⚠ 解压失败，可能需要输入密码"
    echo ""
    echo "请手动执行："
    echo "  ssh -p 25545 root@i-2.gpushare.com 'cd /root/ && unzip -o splits_data.zip -d /hy-tmp/final_test/data/full/'"
fi
