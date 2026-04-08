#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目初始化脚本：自动创建目录结构
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 目录结构定义
DIRECTORIES = [
    "data/stock",
    "data/index",
    "data/macro",
    "data/finance",
    "data/clean",
    "data/combined",
    "output"
]

def create_directories():
    """创建所有需要的目录"""
    for dir_path in DIRECTORIES:
        full_path = os.path.join(PROJECT_ROOT, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

if __name__ == "__main__":
    create_directories()
    print("\nProject directory structure created successfully!")
