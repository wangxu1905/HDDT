#include<hddt.h>

namespace hddt{
const char *status_to_string(status_t status) {
    switch (status) {
    case status_t::SUCCESS:
        return "Succeeded";
    case status_t::ERROR:
        return "Error";
    case status_t::UNSUPPORT:
        return "Unsupported";
    default:
        return "Unknown status";
    }
}
}
