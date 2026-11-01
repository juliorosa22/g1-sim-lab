// LAPTOP STAND-IN for the SDK's publisher/factory: Write only counts.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
namespace unitree { namespace robot {
inline std::atomic<unsigned long long> g_stub_writes{0};
class ChannelFactory {
 public:
  static ChannelFactory* Instance() { static ChannelFactory f; return &f; }
  void Init(int, const std::string&) {}
};
template <typename MSG> class ChannelPublisher {
 public:
  explicit ChannelPublisher(const std::string&) {}
  void InitChannel() {}
  bool Write(const MSG& m, int64_t = 0) {
    g_stub_writes++;
    static FILE* f = std::getenv("STUB_WRITE_LOG") ? std::fopen(std::getenv("STUB_WRITE_LOG"), "w") : nullptr;
    if (f) {  // one line per Write: knee kp, kd, q, motor 0 mode, waist-roll kp/kd, crc
      std::fprintf(f, "%.3f %.3f %.5f %d %.3f %.3f %u\n", (double)m.motor_cmd().at(3).kp(), (double)m.motor_cmd().at(3).kd(), (double)m.motor_cmd().at(3).q(), (int)m.motor_cmd().at(0).mode(),
                   (double)m.motor_cmd().at(13).kp(), (double)m.motor_cmd().at(13).kd(), (unsigned)m.crc());
      std::fflush(f);
    }
    return true;
  }
};
template <typename MSG> using ChannelPublisherPtr = std::shared_ptr<ChannelPublisher<MSG>>;
}}
