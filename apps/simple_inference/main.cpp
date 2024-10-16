#include <iostream>
#include <mpi.h>

class Msg {
public:
  int id;
  int size;
  int *array;

  Msg() : id{0}, array{nullptr}, size{0} {}
  Msg(int id, int *array, int size) : id{id}, array{new int[size]}, size{size} {
    for (int i = 0; i < size; ++i) {
      this->array[i] = array[i];
    }
  }
  ~Msg() { delete[] array; }
  void Print() const;

  static MPI_Datatype CreateMsgType();
};

void Msg::Print() const {
  std::cout << "Message id: " << id << std::endl;
  for (int i = 0; i < size; ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << array[i];
  }
  std::cout << std::endl;
}

MPI_Datatype Msg::CreateMsgType() {
  MPI_Datatype msg_type;
  int blocklengths[3] = {1, 1, 1};                     // 每个字段的个数
  MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT}; // id、size 和 array 指针
  MPI_Aint offsets[3];

  offsets[0] = offsetof(Msg, id);
  offsets[1] = offsetof(Msg, size);
  offsets[2] = offsetof(Msg, array); // 这个持有指向数组的指针

  // 创建结构体类型
  MPI_Type_create_struct(3, blocklengths, offsets, types, &msg_type);
  MPI_Type_commit(&msg_type);

  return msg_type;
}
void Fc_0(MPI_Datatype msgType) {
  // do calculation here

  int data[5] = {1, 2, 3, 4, 5};
  Msg message(1, data, 5);
  std::cout << "Process 0 sending: ";
  message.Print();
  // send message to other process
  MPI_Send(&message, 1, msgType, 1, 0, MPI_COMM_WORLD);
  MPI_Send(message.array, message.size, MPI_INT, 1, 0, MPI_COMM_WORLD);
  std::cout << message.id << " to process 1\n";
}

void Fc_1(MPI_Datatype msgType) {
  Msg received_message;

  MPI_Recv(&received_message, 1, msgType, 0, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  // 接收 array 数据；MPI不支持传递动态分配的数组，需要手动接收
  received_message.array = new int[received_message.size];
  MPI_Recv(received_message.array, received_message.size, MPI_INT, 0, 0,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::cout << "Process 1 received Msg " << received_message.id
            << " from process 0\n";
  received_message.Print();

  // do calculation here
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  MPI_Datatype msg_type = Msg::CreateMsgType();

  //   std::cout << "Hello from processor " << world_rank;
  //   std::cout << " out of " << world_size << " processors." << std::endl;

  if (world_rank == 0) {
    Fc_0(msg_type);
  } else if (world_rank == 1) {
    Fc_1(msg_type);
  }
  MPI_Type_free(&msg_type);
  MPI_Finalize();
  return 0;
}