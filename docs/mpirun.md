---
title: mpi环境一致配置
---

# 卸载已有的openmpi

```shell
sudo apt-get remove --purge openmpi-bin libopenmpi-dev
#清理残留文件
sudo apt-get autoremove
sudo apt-get clean
```

# 安装openmpi

这里以 openmpi 4.1.7为例，下载链接：
https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.7.tar.gz

## 安装

这里安装到`/usr/local`

```shell
# 安装依赖
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev

# 下载
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.7.tar.gz
# 解压
tar -xzvf openmpi-4.1.7.tar.gz

cd openmpi-4.1.7
# 设置安装路径（可选）
./configure --prefix=/usr/local
# 安装
make all
sudo make install
```

![image.png](https://raw.githubusercontent.com/a-c-dream/imgs/master/img/20241119101356.png)

## 环境变量

在`/etc/profile`中添加了下面的内容

```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH  #解决mpirun缺链接库问题
```

## 编译测试用例

```shell
.
├── coll.h
├── hddt.h
├── main.cpp
├── mem.h
├── nn.h
└── p2p.h

```

```shell
g++ -o program main.cpp -lstdc++ -lglog -L <你的glog目录> -I <mpi头文件目录> -lmpi -lrt -I ./
```

## 运行

```shell
#完整测试命令
mpirun -np 2 -pernode --allow-run-as-root \
--host ip1,iP2 \
--mca btl ^openib,ofi \
--mca mtl ^ofi \
--mca pml ob1 \
./program

#最简测试命令
mpirun -np 2 -pernode --allow-run-as-root \
--host ip1,ip2 \
--mca pml ob1 \
./program
```

一些参数的解释：

- **`--mca btl ^openib,ofi`**：
    - `--mca` 是用于设置 Open MPI 参数的选项。`btl` 是 Open MPI 的一个组件，用于控制底层的通信协议（如 TCP, InfiniBand, OpenFabrics 等）。
    - `^openib,ofi` 表示禁用 `openib` 和 `ofi`（OpenFabrics 相关协议）。`^` 表示排除（disable）这些协议，可能是为了避免出现不兼容的问题或调试时避免使用这些协议。
- **`--mca mtl ^ofi`**：

    - `mtl` 代表 Message Transfer Layer，用于控制消息传递的传输层。
    - `^ofi` 表示禁用 OFI（OpenFabrics Interface）传输层协议，通常是因为某些环境下这个协议可能导致不稳定或错误。
- **`--mca pml ob1`**：

  - `pml` 代表 Point-to-Point Message Layer，这是 Open MPI 用来控制进程间点对点通信的组件。
  - `ob1` 是一个基于复制的协议，适用于大多数网络通信，通常用于支持默认的进程间通信。这个参数确保使用 `ob1` 作为 PML 协议。
  ![image-20241127081141422](https://raw.githubusercontent.com/a-c-dream/imgs/master/img/image-20241127081141422.png)

