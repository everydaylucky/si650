# 解压 splits_data.zip 到 data/full 指南

## 方法1: 使用 scp 下载后解压（推荐）

```bash
cd /hy-tmp/final_test
mkdir -p data/full

# 1. 先下载文件
scp -P 25545 root@i-2.gpushare.com:/root/splits_data.zip ./splits_data.zip

# 2. 解压到 data/full
unzip -o splits_data.zip -d data/full/

# 3. 清理（可选）
rm splits_data.zip
```

## 方法2: 直接在远程服务器解压后传输

```bash
# 在远程服务器上解压
ssh -p 25545 root@i-2.gpushare.com 'cd /root/ && unzip -o splits_data.zip -d /tmp/splits_data/'

# 然后使用 scp 传输整个目录
scp -r -P 25545 root@i-2.gpushare.com:/tmp/splits_data/* data/full/
```

## 方法3: 使用 SSH 管道直接解压（如果SSH配置好）

```bash
cd /hy-tmp/final_test
mkdir -p data/full

ssh -p 25545 root@i-2.gpushare.com 'cd /root/ && unzip -p splits_data.zip' | unzip - -d data/full/
```

## 方法4: 如果文件已经在本地

```bash
cd /hy-tmp/final_test
mkdir -p data/full
unzip -o splits_data.zip -d data/full/
```

## 检查解压结果

```bash
ls -lh data/full/
```

## 注意事项

1. 如果SSH需要密码，会提示输入
2. 如果使用密钥认证，确保密钥已配置
3. 解压前确保 data/full 目录有足够空间
