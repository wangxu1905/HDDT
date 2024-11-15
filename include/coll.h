#ifndef HDDT_COLL_H
#define HDDT_COLL_H

#include <ifaddrs.h>
#include <map>
#include <mpi.h>
#include <netdb.h>
#include <p2p.h>

#define HOSTNAME_MAX 256
#define MAX_IP_SIZE 1024

namespace hddt {
class MpiOob {
public:
  int world_size;
  int rank;
  std::map<int, std::string> ip_tables; // 存储rank与ip的映射

  MpiOob(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);

    // 获取本地第一个有效的IP地址
    std::string local_ip = get_local_ip();
    if (local_ip.empty()) {
      throw std::runtime_error("Failed to get local IP address");
    }

    ip_tables[rank] = local_ip;

    // 交换IP地址
    exchange_ip();
  }
  ~MpiOob() { MPI_Finalize(); }

  void exchange_ip() {
    // 使用MPI_Allgather收集所有主机的IP
    int max_ip_len = MAX_IP_SIZE;
    char *all_ips = new char[world_size * max_ip_len];
    char my_ip[MAX_IP_SIZE];
    strcpy(my_ip, ip_tables[rank].c_str());

    MPI_Allgather(my_ip, max_ip_len, MPI_CHAR, all_ips, max_ip_len, MPI_CHAR,
                  MPI_COMM_WORLD);

    for (int i = 0; i < world_size; ++i) {
      char *ip = all_ips + i * max_ip_len;
      ip_tables[i] = std::string(ip);
    }

    delete[] all_ips;
  }
  std::string get_ip(int rank) {
    auto it = ip_tables.find(rank);
    if (it != ip_tables.end()) {
      return it->second;
    } else {
      throw std::out_of_range("Rank not found in ip_tables");
    }
  }

private:
  std::string get_local_ip() {
    struct ifaddrs *ifaddr, *ifa;
    int family, s;
    char host[NI_MAXHOST];

    if (getifaddrs(&ifaddr) == -1) {
      logError("getifaddrs failed.");
      return "";
    }

    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == NULL) {
        continue;
      }

      family = ifa->ifa_addr->sa_family;

      if (family == AF_INET || family == AF_INET6) {
        s = getnameinfo(ifa->ifa_addr,
                        (family == AF_INET) ? sizeof(struct sockaddr_in)
                                            : sizeof(struct sockaddr_in6),
                        host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
        if (s != 0) {
          logDebug("getnameinfo() failed: %s\n", gai_strerror(s));
          continue;
        }

        // Skip loopback addresses
        if (strcmp(host, "127.0.0.1") == 0 || strcmp(host, "::1") == 0) {
          continue;
        }

        // Return the first non-loopback address
        freeifaddrs(ifaddr);
        return std::string(host);
      }
    }

    freeifaddrs(ifaddr);
    return "";
  }
};

//
class AllToALL {
public:
  MpiOob *oob;
  /*When data needs to be transferred between different hosts, network
   * communication technologies such as RDMA are used for efficient data
   * transmission. For data transfers within the same host, GPU memory copy (or
   * similar fast intra-host transfer mechanisms) is employed to achieve faster
   * transfer speeds.*/
  Communicator *comm;

  AllToALL(MpiOob *oob, Communicator *comm) : oob(oob), comm(comm) {}
  ~AllToALL() {}
};

class AllReduce {
  MpiOob *oob; //
  Communicator *comm;

  AllReduce(MpiOob *oob, Communicator *comm) : oob(oob), comm(comm) {}
  ~AllReduce() {}
};
} // namespace hddt

#endif