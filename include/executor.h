#ifndef EXECUTOR_H
#define EXECUTOR_H

#include <hddt.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <atomic>
#include <cuda.h>

namespace hddt {

    // 定义 CUDA 执行器的状态枚举
    enum class ExecutorStatus {
        INITIALIZED,  // 初始化状态
        POSTED,       // 已提交状态
        STARTED,      // 已启动状态
        SHUTDOWN,     // 关闭状态
        SHUTDOWN_ACK  // 关闭确认状态
    };
    
    enum class TaskStatus{
        /* Operation completed successfully */
        OK                              =    0,

        INPROGRESS                      =    1, /*!< Operation is posted and is in progress */

        OPERATION_INITIALIZED           =    2, /*!< Operation initialized but not posted */

        /* Error status codes */
        ERR_NOT_SUPPORTED               =   -1,
        ERR_NOT_IMPLEMENTED             =   -2,
        ERR_INVALID_PARAM               =   -3,
        ERR_NO_MEMORY                   =   -4,
        ERR_NO_RESOURCE                 =   -5,
        ERR_NO_MESSAGE                  =   -6, /*!< General purpose return code without specific error */
        ERR_NOT_FOUND                   =   -7,
        ERR_TIMED_OUT                   =   -8,
        ERR_CUDA                        =   -9,
        ERR_LAST                        = -100,
    };



    // 定义复制任务结构体
    struct TaskCopy {
        void *src;  // 源缓冲区
        void *dst;  // 目标缓冲区
        size_t len;  // 复制长度（以字节为单位）
        TaskStatus status;
        int id;
    };
    /*
    // CUDA 执行器任务操作结构体
    struct TaskOps {
        using TaskPostFunc = status_t (*)(void *executor,
                                     const TaskCopy *task_copy,
                                     TaskCopy **task);  // 提交任务
        using TaskTestFunc = status_t (*)(const TaskCopy *task);  // 测试任务是否完成
        using TaskFinalizeFunc = status_t (*)(TaskCopy *task);  // 释放任务资源

        TaskPostFunc task_post;  // 提交任务
        TaskTestFunc task_test;  // 测试任务是否完成
        TaskFinalizeFunc task_finalize;  // 释放任务资源
    };*/

    // CUDA 执行器类
    class CudaExecutor {
    public:
        CudaExecutor();
        ~CudaExecutor();

        // 启动持久模式的执行器
        status_t Start();

        // 停止持久模式的执行器
        status_t Stop();

        // 发布任务到执行器
        status_t TaskPost(TaskCopy &task_copy);
        
        // 完成任务
        status_t TaskFinalize(TaskCopy *task);

        // 获取执行器状态
        //status_t getStatus() const;

        // 释放执行器资源
        status_t finalize();

        // 核函数启动
        status_t kernelStart(CudaExecutor *executor);

        // 测试任务是否完成
        TaskStatus TaskTest(TaskCopy *task);

	status_t Executor();

    //private:
        //CUcontext *ee_context;  // 执行环境上下文
        //TaskOps ops;  // 任务操作
        mutable std::mutex tasks_lock;  // 任务锁
        ExecutorStatus state;  // 执行器状态
        int pidx;  // 生产者索引
        int cidx;  // 消费者索引
        std::vector<TaskCopy> tasks;  // 任务列表
        ExecutorStatus *dev_state;  // 设备上的执行器状态
        TaskCopy *dev_tasks;  // 设备上的任务列表
        int *dev_pidx;  // 设备上的生产者索引
        int *dev_cidx;  // 设备上的消费者索引
    };

}  // namespace hddt

#endif  // EXECUTOR_H
