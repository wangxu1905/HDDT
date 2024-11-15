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

3. 编译
```
make
```

4. python包(可选)
- CMakeLists.txt 中 set(BUILD_PYTHON_MOD ON)，重新编译整个项目
- 构建并安装包(使用build) `pip install build`
- 构建wheel包 `python -m build`
- 安装wheel包 `pip install dist/xxx.whl`


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
5. glog
    - `sudo apt-get install libgoogle-glog-dev`
6. pybind
    - `git submodule update --init --recursive`