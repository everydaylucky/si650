#!/bin/bash
# 运行所有测试

cd "$(dirname "$0")"
python -m unittest discover tests -v

