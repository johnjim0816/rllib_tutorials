## 1. 运行环境

### 1.1. 创建环境
```bash
conda create -n rllib python=3.8
conda activate rllib
```
### 1.2. 安装rllib
```bash
pip install ray[rllib]==2.0.0 
pip install "gym[atari]" "gym[accept-rom-license]" atari_py
```
### 1.3. 安装TF或者Torch
```bash
pip install tensorflow torch
```
### 1.4. 安装gym
```bash
pip install "gym[atari]" "gym[accept-rom-license]" atari_py
```