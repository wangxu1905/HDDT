#include <iostream>
#include <torch/torch.h>
int main() {
    auto cuda_available = torch::cuda::is_available();
    std::cout << torch::cuda::device_count() << std::endl;
    torch::Device device("cuda:0");
    std::cout << device << std::endl;
    
    std::cout << "---- BASIC AUTOGRAD EXAMPLE 1 ----\n";
    torch::Tensor x = torch::tensor(1.0,torch::requires_grad());
    torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
    torch::Tensor b = torch::tensor(3.0, torch::requires_grad());
    x = x.to(device);
    w = w.to(device);
    b = b.to(device);
    auto y = w*x + b;
    // y.to(device);
    y.backward();
    std::cout << x.cpu().grad() << '\n';  // x.grad() = 2
    std::cout << w.cpu().grad() << '\n';  // w.grad() = 1
    std::cout << b.cpu().grad() << "\n\n";  // b.grad() = 1
    std::cout << "---- BASIC AUTOGRAD EXAMPLE 2 ----\n";
    x = torch::randn({10, 3});
    y = torch::randn({10, 2});
    x = x.to(device);
    y = y.to(device);
    torch::nn::Linear linear(3, 2);
    // std::cout << "w:\n" << linear->weight << '\n';
    // std::cout << "b:\n" << linear->bias << '\n';
    linear->to(device);
    torch::nn::MSELoss criterion;
    torch::optim::SGD optimizer(linear->parameters(), torch::optim::SGDOptions(0.01));
    auto pred = linear->forward(x);
    std::cout << "pred address:" <<  &pred << std::endl;
    // std::cout << "const pred address:" <<  pred.const_data_ptr() << std::endl;
    std::cout << "pred value:" << pred << std::endl;
    auto loss = criterion(pred,y);
    std::cout << "loss value:" <<  loss << std::endl;
    loss.backward();
    std::cout << "dL/dw:\n" << linear->weight.grad() << '\n';
    std::cout << "dL/db:\n" << linear->bias.grad() << '\n';
    // 1 step gradient descent
    optimizer.step();
    std::cout << "dL/dw:\n" << linear->weight.grad() << '\n';
    std::cout << "dL/db:\n" << linear->bias.grad() << '\n';
    pred = linear->forward(x);
    loss = criterion(pred, y);
    std::cout << "Loss after 1 optimization step: " << loss.item<double>() << "\n\n";
}