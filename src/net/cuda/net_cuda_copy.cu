
#include <executor.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <cuda.h>
using namespace cooperative_groups;

#define align_pow2(_n, _p) ((_n) & ((_p) - 1))
#define WARP_SIZE 32
#define vectype uint4  // 假设 vectype 是 uint4
#define COPY_LOOP_UNROLL                  1

namespace hddt {

    // 启动执行器的内核函数
    __global__ void executor_start(ExecutorStatus *state, int *cidx)
    {
        // 初始化消费索引
        *cidx = 0;
        // 设置执行器状态为已启动
        *state = ExecutorStatus::STARTED;
    }

    // 关闭确认的内核函数
    __global__ void executor_shutdown_ack(ExecutorStatus *state)
    {
        // 设置执行器状态为停止确认
        *state = ExecutorStatus::SHUTDOWN_ACK;
    }

    // 任务复制的内核函数
    template<int UNROLL>
    __device__ void executor_copy_task(TaskCopy &task)
    {
        //printf("task start ...\n");
        // 获取任务的长度、源地址和目标地址
        size_t      count     = task.len;
        const char *s1        = reinterpret_cast<const char*>(task.src);
        char       *d1        = reinterpret_cast<char *>(task.dst);

        // 检查源地址和目标地址是否对齐
        if (!(align_pow2((intptr_t)s1, sizeof(vectype)) || align_pow2((intptr_t)d1, sizeof(vectype)))) {
            // 计算线程在 Warp 中的位置和 Warp 的数量
            int            warp      = threadIdx.x / WARP_SIZE;
            int            num_warps = blockDim.x / WARP_SIZE;
            int            idx       = threadIdx.x % WARP_SIZE;
            //printf("threadIdx.x : %d, blockDim.x : %d,warp_size : %d\n",threadIdx.x, blockDim.x,WARP_SIZE);
            // 将源地址和目标地址转换为矢量类型
            const vectype *s4        = reinterpret_cast<const vectype*>(s1);
            vectype       *d4        = reinterpret_cast<vectype*>(d1);
            // 计算要处理的行数
            size_t         num_lines = (count / (WARP_SIZE * UNROLL * sizeof(vectype))) * (WARP_SIZE * UNROLL);
            // 临时存储数组
            vectype        tmp[UNROLL];
            //printf("duiqi ...\n");
            // 主循环，按行处理数据
           // printf("num_warps : %d\n",num_warps);
            //printf("start:%d, end: %d, step: %d\n",warp * WARP_SIZE * UNROLL + idx,num_lines,1);
            //printf("UNROLL:%d ...\n",UNROLL);
            for (size_t line = warp * WARP_SIZE * UNROLL + idx; line < num_lines; line += num_warps * WARP_SIZE * UNROLL) {
                #pragma unroll
                for (int i = 0; i < UNROLL; i++) {
                    tmp[i] = s4[line + WARP_SIZE * i]; // 从源地址读取数据
                }


                #pragma unroll
                for (int i = 0; i < UNROLL; i++) {
                    d4[line + WARP_SIZE * i] = tmp[i]; // 将数据写入目标地址
                }
            }

            // 更新剩余的数据量
            count = count - num_lines * sizeof(vectype);
            if (count == 0) {
                //printf("count = 0\n");
                return; // 如果没有剩余数据，直接返回
            }
               // printf("out for\n");
            // 更新源地址和目标地址
            s4 = s4 + num_lines;
            d4 = d4 + num_lines;
            // 计算剩余的行数
            num_lines = count / sizeof(vectype);
            // 处理剩余的对齐数据
            for (int line = threadIdx.x; line < num_lines; line += blockDim.x) {
                d4[line] = s4[line];
            }
            //printf("out for  2222\n");
            // 更新剩余的数据量
            count = count - num_lines * sizeof(vectype);
            if (count == 0) {
                //printf("count = 0\n");
                return; // 如果没有剩余数据，直接返回
            }

            // 更新源地址和目标地址
            s1 = reinterpret_cast<const char*>(s4 + num_lines);
            d1 = reinterpret_cast<char*>(d4 + num_lines);
        }
        //printf("chulai ...\n");
        // 处理剩余的非对齐数据
        for (size_t line = threadIdx.x; line < count; line += blockDim.x) {
            d1[line] = s1[line];
        }
        printf("......task finished .....\n");
    }

        __global__ void executor_kernel(CudaExecutor *executor, int q_size) {
            //printf("1\n");
            // 计算当前线程块（工作单元）的 ID 和总线程块数量
            const uint32_t worker_id = blockIdx.x;
            const uint32_t num_workers = gridDim.x;
            // 判断当前线程是否为每个线程块中的第一个线程（master 线程）
            bool is_master = (threadIdx.x == 0) ? true : false;
            // 本地变量初始化
            int cidx_local, pidx_local;
            volatile int *pidx, *cidx;
            TaskCopy *tasks;
            __shared__ TaskCopy args; // 共享内存中存储任务参数
            __shared__ bool worker_done; // 标志位，指示任务是否完成
            //printf("2\n");

            // 如果是 master 线程，则进行初始化工作
            if (is_master) {
                // 初始化本地变量
                cidx_local = worker_id;
                //printf("3\n");
                pidx = executor->dev_pidx;
                cidx = executor->dev_cidx;
                tasks = executor->dev_tasks;

                // 检查 pidx 和 cidx 的值
                // cidx_loacl : 0    pidx : 1      cidx : 0
                printf("master : %d, *pidx: %d, *cidx: %d\n", cidx_local, *pidx, *cidx);

                // 确保 pidx 和 cidx 的值在合理范围内
                if (*pidx < 0 || *pidx >= q_size || *cidx < 0 || *cidx >= q_size) {
                    printf("Invalid pidx or cidx: *pidx: %d, *cidx: %d\n", *pidx, *cidx);
                    return;
                }
            }

            // 初始化共享内存中的标志位
            worker_done = false;
            // 同步所有线程
            __syncthreads();

            //printf("4\n");

            // 主循环，持续执行直到所有任务完成
            while (true) {
                if (is_master) { // 只有 master 线程进行任务获取和状态更新
                    // 等待到当前线程块对应的下一个任务
                    // cidx_loacl : 0    pidx : 1      cidx : 0   num_worker: 1 work_id: 0
                    printf("cidx : %d,num_workers : %d, worker_id : %d\n",*cidx,num_workers,worker_id);
                    while ((*cidx % num_workers) != worker_id);
                    // 检查是否有新任务可用
                    pidx_local = *pidx;
                    // 检查是否为结束信号
                    //1   1
                    worker_done = (pidx_local == (*cidx));
                    (*cidx)++; // 更新消费索引
                    if (!worker_done) { // 如果不是结束信号，则获取任务
                        args = tasks[cidx_local];
                        printf("cidx_local : %d\n",cidx_local);
                    }
                }
                // 确保所有线程都完成了状态更新
                __syncthreads();

                // 如果任务已完成，根据启动方式执行不同的清理工作
                if (worker_done) {
                    return; // 结束内核执行
                }

                // 处理数据复制任务
                if (!worker_done) {
                    executor_copy_task<COPY_LOOP_UNROLL>(args);
                }

                // 确保所有线程都完成了任务处理
                __syncthreads();
                // 确保所有线程都看到了最新的内存状态
                __threadfence_system();

                if (is_master) { // 任务完成后，由 master 线程更新任务状态
                    tasks[(cidx_local)].status = TaskStatus::OK;
                    printf("tasks OK.....\n");
                    // 计算下一个任务的位置
                    cidx_local = (cidx_local + num_workers) % q_size;
                }

                __syncthreads();
            }
        }

    extern "C" {
    // 启动持久内核执行器的函数
        status_t CudaExecutor::kernelStart(CudaExecutor *executor)
        {
            if (executor == nullptr) {
                std::cerr << "Executor is null!" << std::endl;
                return status_t::ERROR;
            }

            // 获取 CUDA 流
            //cudaStream_t stream = (cudaStream_t)eee_context;

            // 从配置中读取执行器的参数
            int nb = 1; // 工作单元数量
            int nt = 64; // 每个工作单元的线程数
            int q_size = 1024; // 最大任务队列大小

            std::cout << "in kernelStart ..." << std::endl;
            executor_start<<<1, 1, 0>>>(executor->dev_state, executor->dev_cidx);
            // 启动主要任务处理内核
            
            executor_kernel<<<nb, nt, 0>>>(executor, q_size);
            // 启动关闭确认内核
            cudaDeviceSynchronize();
            std::cout << "executor finished ..." << std::endl;
            executor_shutdown_ack<<<1, 1, 0>>>(executor->dev_state);
            

            // 检查 CUDA 错误
            cudaError_t cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << std::endl;
                return status_t::ERROR;
            }

            std::cout << "kernel finished ..." << std::endl;

            return status_t::SUCCESS;
        }
    }

}  // namespace hddt
