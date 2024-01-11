#include <pybind11/pybind11.h>
#include "dual_c.hpp"  // Include the header file with the DualNumber class

namespace py = pybind11;

PYBIND11_MODULE(dual_number, m) {
    py::class_<DualNumber>(m, "DualNumber")
        .def(py::init<double, double>())
        .def("__mul__", &DualNumber::operator*)
        .def("__rmul__", [](const DualNumber& self, double num) { return self * num; })
        .def("__add__", &DualNumber::operator+)
        .def("__radd__", &DualNumber::operator+)
        .def("__sub__", &DualNumber::operator-)
        .def("__rsub__", &DualNumber::operator-)
        .def("__truediv__", &DualNumber::operator/)
        .def("__rtruediv__", &DualNumber::operator/)
        .def("__neg__", &DualNumber::operator-)
        .def("__pow__", &DualNumber::operator^)
        .def("sin", &DualNumber::sin)
        .def("cos", &DualNumber::cos)
        .def("tan", &DualNumber::tan)
        .def("log", &DualNumber::log)
        .def("exp", &DualNumber::exp);
}
