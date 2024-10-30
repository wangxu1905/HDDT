#include<iostream>
#include<hddt.hpp>

int main() {
    hddt_status_t ret;
    ret = hddt::init_gpu_driver();
    if(ret == hddt_status_t::SUCCESS) {
        logInfo("init gpu driver success!");
    }else{
        logError("init gpu driver failed!");
    }
    return 0;
}