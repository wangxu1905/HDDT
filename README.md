# HDDT
Distrubuted DNN Training on Heterogeneous GPUs

1. 创建编译目录
```
mkdir build && cd build
```

2. 生成makefile
```
cmake ..
```

支持的参数
```
-DBUILD_STATIC_LIB=ON # 开启静态库编译
```

2. 编译
make


# 环境依赖
1. 计算库驱动 CUDA/DTK/CNRT etc.
2. openMPI
    `sudo apt install openmpi-bin openmpi-common libopenmpi-dev`
3. Miniconda
    - `https://docs.anaconda.com/miniconda/`
    - `conda create -n py310 python=3.10`
4. pytorch
    - `pip3 install torch torchvision torchaudio`
    - `python -c "import torch; print(torch.cuda.is_available())"`

for ubuntu:
- `apt-get install libpci-dev`
- `apt-get install libgtest-dev` // for test