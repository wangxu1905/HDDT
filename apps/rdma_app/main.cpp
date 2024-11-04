#include <hddt.h>
#include <iostream>

int main() {
  hddt::status_t ret;

  ret = hddt::init_gpu_driver(1);
  if (ret == hddt::status_t::SUCCESS) {
    hddt::logInfo("init gpu driver success!");
  } else {
    hddt::logError("init gpu driver failed!");
  }
  return 0;
}