#include "executor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cuda.h>
#include <thread>
#include <chrono>

// 辅助函数：检查 CUDA 调用的结果
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main() {
    hddt::CudaExecutor executor; 
    // 启动执行器
    hddt::status_t start_status = executor.Start();
    if (start_status != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to start the executor: " << static_cast<int>(start_status) << std::endl;
        return 1;
    }

    // 准备测试任务
    size_t task1_size = 4;
    size_t task2_size = 8;

    // 分配主机内存
    char *host_src1 = new char[task1_size];
    char *host_dst1 = new char[task1_size];
    char *host_src2 = new char[task2_size];
    char *host_dst2 = new char[task2_size];

    // 初始化源数据
    for (size_t i = 0; i < task1_size; ++i) {
        host_src1[i] = 'B' + i;
        std::cout << host_src1[i] << " " << std::endl;
    }

        std::cout << std::endl;    std::cout << std::endl;    std::cout << std::endl;    std::cout << std::endl;
    for (size_t i = 0; i < task2_size; ++i) {
        host_src2[i] = 'a' + i;
        std::cout << host_src2[i] << " " << std::endl;
    }
    std::cout << std:: endl;
    

    // 分配设备内存
    char *dev_src1, *dev_dst1, *dev_src2, *dev_dst2;
    cudaError_t err;

    err = cudaMalloc(&dev_src1, task1_size);
    checkCudaError(err, "cudaMalloc failed for dev_src1");

    err = cudaMalloc(&dev_dst1, task1_size);
    checkCudaError(err, "cudaMalloc failed for dev_dst1");

    err = cudaMalloc(&dev_src2, task2_size);
    checkCudaError(err, "cudaMalloc failed for dev_src2");

    err = cudaMalloc(&dev_dst2, task2_size);
    checkCudaError(err, "cudaMalloc failed for dev_dst2");

    //std::cout << "memory inited ...." << std::endl;

    // 将数据从主机复制到设备
    err = cudaMemcpy(dev_src1, host_src1, task1_size, cudaMemcpyHostToDevice);
    checkCudaError(err, "cudaMemcpy failed for dev_src1");

    err = cudaMemcpy(dev_src2, host_src2, task2_size, cudaMemcpyHostToDevice);
    checkCudaError(err, "cudaMemcpy failed for dev_src2");

    //std::cout << "data copy to device finished ...." << std::endl;

    // 创建任务结构体
    hddt::TaskCopy task1 = {dev_src1, dev_dst1, task1_size, hddt::TaskStatus::OPERATION_INITIALIZED};
    hddt::TaskCopy task2 = {dev_src2, dev_dst2, task2_size, hddt::TaskStatus::OPERATION_INITIALIZED};

    hddt::status_t post_status1 = executor.TaskPost(task1);
    if (post_status1 != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to post task1: " << static_cast<int>(post_status1) << std::endl;
    }

    hddt::status_t post_status2 = executor.TaskPost(task2);
    if (post_status2 != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to post task2: " << static_cast<int>(post_status2) << std::endl;
    }

    std::cout << "tasks post finished ...." << std::endl;

    executor.Executor();
    //cudaDeviceSynchronize();
    // 等待任务完成
    hddt::TaskStatus status1 = hddt::TaskStatus::INPROGRESS;
    while (status1 != hddt::TaskStatus::OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        cudaMemcpy(&status1, &(executor.dev_tasks[task1.id].status), sizeof(hddt::TaskStatus), cudaMemcpyDeviceToHost);
        
    }

    hddt::TaskStatus status2 = hddt::TaskStatus::INPROGRESS;
    while (status2 != hddt::TaskStatus::OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        cudaMemcpy(&status2, &(executor.dev_tasks[task2.id].status), sizeof(hddt::TaskStatus), cudaMemcpyDeviceToHost);
        
    }
/*
    // 完成任务
    hddt::status_t finalize_status1 = executor.TaskFinalize(&task1);
    if (finalize_status1 != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to finalize task1: " << static_cast<int>(finalize_status1) << std::endl;
    }
*/ 
    /*hddt::status_t finalize_status2 = executor.TaskFinalize(&task2);
    if (finalize_status2 != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to finalize task2: " << static_cast<int>(finalize_status2) << std::endl;
    }*/

    std::cout << "tasks finished ...." << std::endl;

    // 将结果从设备复制回主机
    err = cudaMemcpy(host_dst1, dev_dst1, task1_size, cudaMemcpyDeviceToHost);
    checkCudaError(err, "cudaMemcpy failed for host_dst1");

    err = cudaMemcpy(host_dst2, dev_dst2, task2_size, cudaMemcpyDeviceToHost);
    checkCudaError(err, "cudaMemcpy failed for host_dst2");

     //验证结果
    for (size_t i = 0; i < task1_size; ++i) {
    std::cout << host_dst1[i] << " " << host_src1[i] << std::endl;
        assert(host_dst1[i] == host_src1[i]);
    }

    std::cout << std::endl;    std::cout << std::endl;    std::cout << std::endl;    std::cout << std::endl;
  for (size_t i = 0; i < task2_size; ++i) {
        std::cout << host_dst2[i] << " " << host_src2[i] << std::endl;
        assert(host_dst2[i] == host_src2[i]);
    }

    delete[] host_src1;
    delete[] host_dst1;
    delete[] host_src2;
    delete[] host_dst2;

    if (dev_src1) cudaFree(dev_src1);
    if (dev_dst1) cudaFree(dev_dst1);
    if (dev_src2) cudaFree(dev_src2);
    if (dev_dst2) cudaFree(dev_dst2);

    // 停止执行器
    hddt::status_t stop_status = executor.Stop();
    if (stop_status != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to stop the executor: " << static_cast<int>(stop_status) << std::endl;
        return 1;
    }

    std::cout << "All tasks completed successfully." << std::endl;

    return 0;
}
