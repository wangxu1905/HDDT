#include "hddt.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(hddt, m) {
  // export function
  m.def("add", &hddt::add, "A function which adds two numbers");

  // export class
  py::class_<hddt::Pet>(m, "Pet")
      .def(py::init<const std::string &>())
      .def("setName", &hddt::Pet::setName)
      .def("getName", &hddt::Pet::getName);
}