#include "executor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cuda.h>
#include <thread>
#include <chrono>

namespace hddt {

    CudaExecutor::CudaExecutor() 
        : state(ExecutorStatus::INITIALIZED), pidx(0), cidx(0) {
        // 初始化设备上的状态和任务列表
        cudaMalloc((void**)&dev_state, sizeof(ExecutorStatus));
        cudaMalloc((void**)&dev_tasks, sizeof(TaskCopy) * 1024);  // 假设 TASK_QUEUE_SIZE 为 1024
        cudaMalloc((void**)&dev_pidx, sizeof(int));
        cudaMalloc((void**)&dev_cidx, sizeof(int));

        // 初始化状态和索引
        cudaMemcpy(dev_state, &state, sizeof(ExecutorStatus), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_pidx, &pidx, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_cidx, &cidx, sizeof(int), cudaMemcpyHostToDevice);
    }

    CudaExecutor::~CudaExecutor() {
        // 释放设备上的资源
        cudaFree(dev_state);
        cudaFree(dev_tasks);
        cudaFree(dev_pidx);
        cudaFree(dev_cidx);
    }

    status_t CudaExecutor::Start() {

        // 断言执行器状态为已初始化
        assert(state == ExecutorStatus::INITIALIZED);

        // 打印调试信息
        std::cout << "executor start, eee: " << this << std::endl;

        // 设置执行器上下文和状态
        //ee_context = ee_context;  // 这里假设 ee_context 已经在构造函数中设置
        state = ExecutorStatus::POSTED;
        pidx = 0;


        // 启动持久内核
        status_t status = kernelStart(this);
        if (status != status_t::SUCCESS) {
            // 如果启动内核失败，记录错误并返回
            std::cerr << "failed to launch executor kernel" << std::endl;
            return status;
        }

        /*
        // 设置执行器的操作函数
        ops.task_post = TaskPost;
        ops.task_test = TaskTest;
        ops.task_finalize = TaskFinalize;
        */
        return status_t::SUCCESS;
    }

    status_t CudaExecutor::Stop() {

        // 调试信息
        std::cout << "executor stop, eee: " << this << std::endl;

        // 断言执行器状态为已启动或已完成确认
        assert(state == ExecutorStatus::POSTED || state == ExecutorStatus::SHUTDOWN);

        // 设置执行器状态为停止
        state = ExecutorStatus::SHUTDOWN;
        pidx = -1; // 设置生产者索引为 -1 表示停止

        // 等待执行器确认停止
        while (state != ExecutorStatus::SHUTDOWN_ACK) { }

        // 重置执行器上下文和状态
        //ee_context = nullptr;
        state = ExecutorStatus::INITIALIZED;

        return status_t::SUCCESS;
    }

    status_t CudaExecutor::TaskPost(TaskCopy &task_copy) {
        if (state != ExecutorStatus::POSTED) {
           return status_t::ERROR; 
        }

        // 获取最大任务数
        int max_tasks = 1024;  // 假设 TASK_QUEUE_SIZE 为 1024

        // 初始化任务结构体
        TaskCopy *ee_task = new TaskCopy(); // 假设我们直接在堆上分配任务结构体
        if (!ee_task) {
            return status_t::ERROR; // 如果没有可用的任务结构体，返回内存不足错误
        }

        // 初始化任务结构体
        ee_task->src = task_copy.src;
        ee_task->dst = task_copy.dst;
        ee_task->len = task_copy.len;
        ee_task->status = TaskStatus::INPROGRESS;

        // 保存任务到任务队列
        tasks[pidx] = *ee_task;
        pidx = (pidx + 1) % max_tasks;  // 更新生产者索引

        // 确保内存写入顺序
        cudaDeviceSynchronize();

        // 打印调试信息
        std::cout << "executor task post, eee: " << this << std::endl;

        //*task = ee_task; // 返回任务指针
        return status_t::SUCCESS;
    }

    status_t CudaExecutor::TaskFinalize(TaskCopy *task) {

        // 断言任务状态为成功
        assert(task->status == TaskStatus::OK);

        return status_t::SUCCESS; 
    }
/*
    ExecutorStatus CudaExecutor::getStatus() const {
        return this->;  // 假设 UCC_OK 为 0
    */

    status_t CudaExecutor::finalize() {
        // 释放资源的逻辑
        return status_t::SUCCESS;  // 假设 UCC_OK 为 0
    }


    TaskStatus CudaExecutor::TaskTest(TaskCopy *task) {

        // 如果任务状态不是进行中，直接返回
        if (task->status != TaskStatus::INPROGRESS) {
            return task->status;
        }

        // 检查 CUDA 错误
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            task->status = TaskStatus::ERR_CUDA; // 如果有 CUDA 错误，设置任务状态为错误
            return task->status;
        }

        // 所有子任务完成，设置任务状态为成功
        task->status = TaskStatus::OK;

        return task->status;
    }

    int main() {
        // 初始化 CUDA 驱动 API
        /*CUresult cu_result = cuInit(0);
        if (cu_result != CUDA_SUCCESS) {
            const char *error_str;
            cuGetErrorString(cu_result, &error_str);
            std::cerr << "Failed to initialize CUDA driver API: " << error_str << std::endl;
            return 1;
        }

        // 获取当前的 CUDA 上下文
        CUcontext ee_context = nullptr;
        cu_result = cuCtxGetCurrent(&ee_context);
        if (cu_result != CUDA_SUCCESS || ee_context == nullptr) {
            const char *error_str;
            cuGetErrorString(cu_result, &error_str);
            std::cerr << "Failed to get current CUDA context: " << error_str << std::endl;
            return 1;
        }*/

        // 创建执行器对象
        //hddt::CudaExecutor executor(&ee_context);  // 假设 ee_context 为 nullptr
        hddt::CudaExecutor executor; 
        // 启动执行器
        hddt::status_t start_status = executor.Start();
        if (start_status != hddt::status_t::SUCCESS) {
            std::cerr << "Failed to start the executor: " << static_cast<int>(start_status) << std::endl;
            return 1;
        }

        // 准备测试任务
        size_t task1_size = 1024;
        size_t task2_size = 2048;

        // 分配主机内存
        char *host_src1 = new char[task1_size];
        char *host_dst1 = new char[task1_size];
        char *host_src2 = new char[task2_size];
        char *host_dst2 = new char[task2_size];

        // 初始化源数据
        for (size_t i = 0; i < task1_size; ++i) {
            host_src1[i] = static_cast<char>(i % 256);
        }
        for (size_t i = 0; i < task2_size; ++i) {
            host_src2[i] = static_cast<char>(i % 256);
        }

        // 分配设备内存
        char *dev_src1, *dev_dst1, *dev_src2, *dev_dst2;
        cudaMalloc(&dev_src1, task1_size);
        cudaMalloc(&dev_dst1, task1_size);
        cudaMalloc(&dev_src2, task2_size);
        cudaMalloc(&dev_dst2, task2_size);

        // 将数据从主机复制到设备
        cudaMemcpy(dev_src1, host_src1, task1_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_src2, host_src2, task2_size, cudaMemcpyHostToDevice);

        // 创建任务结构体
        hddt::TaskCopy task1 = {dev_src1, dev_dst1, task1_size, hddt::TaskStatus::OPERATION_INITIALIZED};
        hddt::TaskCopy task2 = {dev_src2, dev_dst2, task2_size, hddt::TaskStatus::OPERATION_INITIALIZED};

        hddt::status_t post_status1 = executor.TaskPost(task1);
        hddt::status_t post_status2 = executor.TaskPost(task2);

        if (post_status1 != hddt::status_t::SUCCESS) {
            std::cerr << "Failed to post task1: " << static_cast<int>(post_status1) << std::endl;
            return 1;
        }

        if (post_status2 != hddt::status_t::SUCCESS) {
            std::cerr << "Failed to post task2: " << static_cast<int>(post_status2) << std::endl;
            return 1;
        }

        // 等待任务完成
        while (executor.TaskTest(&task1) != hddt::TaskStatus::OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        while (executor.TaskTest(&task2) != hddt::TaskStatus::OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // 完成任务
        hddt::status_t finalize_status1 = executor.TaskFinalize(&task1);
        hddt::status_t finalize_status2 = executor.TaskFinalize(&task2);

        if (finalize_status1 != hddt::status_t::SUCCESS) {
            std::cerr << "Failed to finalize task1: " << static_cast<int>(finalize_status1) << std::endl;
            return 1;
        }

        if (finalize_status2 != hddt::status_t::SUCCESS) {
            std::cerr << "Failed to finalize task2: " << static_cast<int>(finalize_status2) << std::endl;
            return 1;
        }

        // 将结果从设备复制回主机
        cudaMemcpy(host_dst1, dev_dst1, task1_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_dst2, dev_dst2, task2_size, cudaMemcpyDeviceToHost);

        // 验证结果
        for (size_t i = 0; i < task1_size; ++i) {
            assert(host_dst1[i] == host_src1[i]);
        }
        for (size_t i = 0; i < task2_size; ++i) {
            assert(host_dst2[i] == host_src2[i]);
        }

        // 释放内存
        delete[] host_src1;
        delete[] host_dst1;
        delete[] host_src2;
        delete[] host_dst2;
        cudaFree(dev_src1);
        cudaFree(dev_dst1);
        cudaFree(dev_src2);
        cudaFree(dev_dst2);

        // 停止执行器
        hddt::status_t stop_status = executor.Stop();
        if (stop_status != hddt::status_t::SUCCESS) {
            std::cerr << "Failed to stop the executor: " << static_cast<int>(stop_status) << std::endl;
            return 1;
        }

        std::cout << "All tasks completed successfully." << std::endl;

        return 0;
    }

}  // namespace hddt
