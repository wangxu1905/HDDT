---
title: mpirun
---



mpioob是什么？？？
对于一组进程的一个管理，这一组进程要相互交换ip ，rank等
这里的rank可以理解为id

机子：
海光1，寒武纪2,232,233

可能的原因：海光1和寒武纪2的mpirun路径各种问题。

可能的方法：
重装海光1和寒武纪2的mpirun，
也可以试试232和233通不通

# 编译
```shell
.
├── coll.h
├── hddt.h
├── main.cpp
├── mem.h
├── nn.h
└── p2p.h

```

命令
寒武纪2
```shell
g++ -o program main.cpp -lstdc++ -lglog -L /usr/local/lib/python3.10/dist-packages/torch/include/c10/util/ -I ./ -lmpi -lrt
```

海光1
```shell
g++ -o program main.cpp -lstdc++ -lglog -L /usr/lib/x86_64-linux-gnu/ -I ./ -lmpi -lrt 
```

233
```shell
g++ -o program main.cpp -lstdc++ -lglog -L /usr/lib/x86_64-linux-gnu/ -I      /usr/mpi/gcc/openmpi-4.1.7a1/include/ -lmpi -lrt -I ./

```

`/usr/local/lib/python3.10/dist-packages/torch/include/c10/util/`要替换成自己的glog目录
可以通过下面的命令查找
```shell
find /usr/local -name "libglog*"
find /usr/local -name "*glog.h"
```

寒武纪2本地可以正常运行


```shell
mpirun -np 2 -pernode \
--allow-run-as-root \
-host 10.102.0.233,10.102.0.235 \
-mca btl_tcp_if_include eno1np0 \
./program 


mpirun -np 2 -pernode \
--allow-run-as-root \
-host 10.102.0.233,10.102.0.235 \
./program 


ens58f1
```


在寒武纪2上启动
![image.png](https://raw.githubusercontent.com/a-c-dream/imgs/master/img/20241119092339.png)


海光1上启动
![image.png](https://raw.githubusercontent.com/a-c-dream/imgs/master/img/20241119092613.png)
初步怀疑是海光1上的openmpi有问题，`/usr/local/bin`目录下什么都没有

# 重装海光1 openmpi
因为寒武纪2 openmpi版本是4.2.2，海光1也装4.2.2，安装目录是`/usr/local/bin`
没有找到4.2.2，安装了4.1.7
https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.7.tar.gz

## 卸载
```shell
sudo apt-get remove --purge openmpi-bin libopenmpi-dev
#清理残留文件
sudo apt-get autoremove
sudo apt-get clean
```

## 安装

```shell
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev


wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.7.tar.gz

tar -xzvf openmpi-4.1.7.tar.gz

cd openmpi-4.1.7
./configure --prefix=/usr/local
make all
sudo make install
```

![image.png](https://raw.githubusercontent.com/a-c-dream/imgs/master/img/20241119101356.png)


如果233和海光是缺库`libefa.so.1`
寒武纪和其他是缺文件`hydra_pmi_proxy`


```shell
#这个命令可以通
mpirun -np 2 -pernode --allow-run-as-root \
--host 10.102.0.240,10.102.0.235 \
--mca btl ^openib,ofi \
--mca mtl ^ofi \
--mca pml ob1 \
./program

#最小能通命令
mpirun -np 2 -pernode --allow-run-as-root \
--host 10.102.0.233,10.102.0.235 \
--mca pml ob1 \
./program


mpirun -np 2 -pernode --allow-run-as-root \
--host 10.102.0.240,10.102.0.235 \
--mca btl ^openib,ofi \
--mca mtl ^ofi \
--mca pml ob1 \
-x LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
./program
```

- **`--mca btl ^openib,ofi`**：
  
    - `--mca` 是用于设置 Open MPI 参数的选项。`btl` 是 Open MPI 的一个组件，用于控制底层的通信协议（如 TCP, InfiniBand, OpenFabrics 等）。
    - `^openib,ofi` 表示禁用 `openib` 和 `ofi`（OpenFabrics 相关协议）。`^` 表示排除（disable）这些协议，可能是为了避免出现不兼容的问题或调试时避免使用这些协议。
- **`--mca mtl ^ofi`**：
  
    - `mtl` 代表 Message Transfer Layer，用于控制消息传递的传输层。
    - `^ofi` 表示禁用 OFI（OpenFabrics Interface）传输层协议，通常是因为某些环境下这个协议可能导致不稳定或错误。
- **`--mca pml ob1`**：
  
    - `pml` 代表 Point-to-Point Message Layer，这是 Open MPI 用来控制进程间点对点通信的组件。
    - `ob1` 是一个基于复制的协议，适用于大多数网络通信，通常用于支持默认的进程间通信。这个参数确保使用 `ob1` 作为 PML 协议。
  ![image.png](https://raw.githubusercontent.com/a-c-dream/imgs/master/img/20241119140926.png)
  这个测试，寒武纪2和233还有海光1都不通，目前是怀疑寒武纪2的openmpi有问题（它和233还有海光的openmpi不一样）
  233和海光1是通的

下面重装寒武纪2的openmpi

```shell
sudo apt-get remove --purge openmpi*
sdu@DisAI-Cambricon-2:/usr/local$ sudo rm -rf mpich

```

安装过程和前面的一样
![image.png](https://raw.githubusercontent.com/a-c-dream/imgs/master/img/20241119150453.png)

安装完后报错找不到链接库，在`bashrc`中添加了下面的内容
```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```
修改后原来的mpi又出来了
![](https://raw.githubusercontent.com/a-c-dream/imgs/master/img/20241119152541.png)
对`bashrc`做了下面的修改（红色表示删除，绿色表示添加）
![image.png](https://raw.githubusercontent.com/a-c-dream/imgs/master/img/20241119153026.png)
上面的操作并不能让所有用户都解决这个问题
在`/etc/profile`中添加了下面的内容
```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH  #解决mpirun却链接库问题
```


```shell

mpirun -np 2 -pernode --allow-run-as-root \
--host 10.102.0.240,10.102.0.233 \
--mca btl ^openib,ofi \
--mca mtl ^ofi \
--mca pml ob1 \
-x LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
./program

mpirun -np 2 -pernode --allow-run-as-root \
--host 10.102.0.240,10.102.0.233 \
--mca btl ^openib,ofi \
--mca mtl ^ofi \
--mca pml ob1 \
-x PMIX_MCA_gds=hash \
./program

export PMIX_MCA_gds=^ds12


mpirun -np 2 -pernode --allow-run-as-root \
--host 10.102.0.240,10.102.0.233 \
--mca pml ob1 \
./program
```


目前除寒武纪2和233不通，其他都能互通。
寒武纪2和233的问题

![image.png](https://raw.githubusercontent.com/a-c-dream/imgs/master/img/20241120175904.png)
