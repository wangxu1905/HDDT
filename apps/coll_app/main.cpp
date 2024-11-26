#include <coll.h>
#include <iostream>

using namespace hddt;

int main(int argc, char *argv[]) {
  MpiOob *oob = new MpiOob(argc, argv);

  logInfo("current rank: %d, ip: %s", oob->rank,
          oob->get_ip(oob->rank).c_str());
  for (int i = 0; i < oob->world_size; ++i) {
    logInfo("rank %d, oob: %s", i, oob->get_ip(i).c_str());
  }

  return 0;
}

// sudo mpirun -np 2 -host ip1,ip2 ./coll_app