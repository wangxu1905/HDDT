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
        cudaError_t err;
        err = cudaMalloc((void**)&dev_state, sizeof(ExecutorStatus));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate dev_state: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA allocation failed");
        }
        err = cudaMalloc((void**)&dev_tasks, sizeof(TaskCopy) * 1024);  // 假设 TASK_QUEUE_SIZE 为 1024
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate dev_tasks: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA allocation failed");
        }
        err = cudaMalloc((void**)&dev_pidx, sizeof(int));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate dev_pidx: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA allocation failed");
        }
        err = cudaMalloc((void**)&dev_cidx, sizeof(int));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate dev_cidx: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA allocation failed");
        }

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

        /*
        // 设置执行器的操作函数
        ops.task_post = TaskPost;
        ops.task_test = TaskTest;
        ops.task_finalize = TaskFinalize;
        */
        return status_t::SUCCESS;
    }

    status_t CudaExecutor::Executor() {
        // 启动持久内核
        status_t status = kernelStart(this);
        if (status != status_t::SUCCESS) {
            // 如果启动内核失败，记录错误并返回
            std::cerr << "failed to launch executor kernel" << std::endl;
            return status;
        }
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
        //while (state != ExecutorStatus::SHUTDOWN_ACK) { }

        // 重置执行器上下文和状态
        //ee_context = nullptr;
        state = ExecutorStatus::INITIALIZED;

        return status_t::SUCCESS;
    }

    status_t CudaExecutor::TaskPost(TaskCopy &task_copy) {
        if (state != ExecutorStatus::POSTED) {
           return status_t::ERROR; 
        }
        //std::cout << "task post in " << std::endl;
        // 获取最大任务数
        int max_tasks = 1024;  // 假设 TASK_QUEUE_SIZE 为 1024

        // 初始化任务结构体
        TaskCopy *ee_task = new TaskCopy(); // 假设我们直接在堆上分配任务结构体
        if (!ee_task) {
            return status_t::ERROR; // 如果没有可用的任务结构体，返回内存不足错误
        }

        //std::cout << "task new"  << std::endl;

        // 初始化任务结构体
        ee_task->src = task_copy.src;
        ee_task->dst = task_copy.dst;
        ee_task->len = task_copy.len;
        ee_task->status = TaskStatus::INPROGRESS;
        ee_task->id = task_copy.id = pidx;

        //std::cout << "task fuzhi"  << std::endl;
        //std::cout << "pidx : " << pidx << "    dev_tasks" << sizeof(dev_tasks) << std::endl;
        // 保存任务到任务队列
        cudaMemcpy(&dev_tasks[pidx], ee_task, sizeof(TaskCopy), cudaMemcpyHostToDevice);
        //dev_tasks[pidx] = *ee_task;
        pidx = (pidx + 1) % max_tasks;  // 更新生产者索引
        //cudaMemcpy(&dev_pidx, &pidx, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_pidx, &pidx, sizeof(int), cudaMemcpyHostToDevice);

        //std::cout << "task queue"  << std::endl;

        // 确保内存写入顺序
        cudaDeviceSynchronize();

        // 打印调试信息
        //std::cout << "executor task post, eee: " << this << std::endl;

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
        return task->status;
    }
/*
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
    }*/

}  // namespace hddt
