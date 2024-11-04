#include <hddt.h>
#include <cstdarg>
#include <cstdio>

namespace hddt {

void logError(const char *format, ...) {
    va_list args;
    va_start(args, format);
    std::cerr << "[ERROR] ";
    vfprintf(stderr, format, args);
    std::cerr << std::endl;
    va_end(args);
}

void logDebug(const char *format, ...) {
    va_list args;
    va_start(args, format);
    std::cerr << "[DEBUG] ";
    vfprintf(stderr, format, args);
    std::cerr << std::endl;
    va_end(args);
}

void logInfo(const char *format, ...) {
    va_list args;
    va_start(args, format);
    std::cerr << "[INFO] ";
    vfprintf(stderr, format, args);
    std::cerr << std::endl;
    va_end(args);
}

} // namespace hddt
