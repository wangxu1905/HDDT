#include <hddt.h>

namespace hddt {

void logError(const char *message) {
  std::cerr << "[ERROR] " << message << std::endl;
}
void logDebug(const char *message) {
  std::cerr << "[DEBUG] " << message << std::endl;
}
void logInfo(const char *message) {
  std::cerr << "[INFO] " << message << std::endl;
}

} // namespace hddt
