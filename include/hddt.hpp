#ifndef HDDT_H
#define HDDT_H

#include<pthread.h>
#include<iostream>

enum class hddt_status_t { SUCCESS, ERROR };

void logError(const char* message) {
    std::cerr << "[ERROR] " << message << std::endl;
}
void logDebug(const char* message) {
    std::cerr << "[DEBUG] " << message << std::endl;
}
void logInfo(const char* message) {
    std::cerr << "[INFO] " << message << std::endl;
}

namespace hddt {
/*
gpu driver init
*/
hddt_status_t init_gpu_driver();
} // namespace hddt

#endif