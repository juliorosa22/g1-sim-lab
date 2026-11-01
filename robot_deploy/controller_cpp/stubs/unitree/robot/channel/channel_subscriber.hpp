// LAPTOP STAND-IN for the SDK's subscriber: a LowState_ subscriber feeds a quiet, still, upright robot at 1 kHz, with a valid checksum.
#pragma once
#include <cstdio>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include "unitree/idl/hg/LowCmd_.hpp"
#include "unitree/idl/hg/LowState_.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"
namespace unitree { namespace robot {
inline uint32_t stub_crc(uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0, data = 0, CRC32 = 0xFFFFFFFF; const uint32_t poly = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) { xbit = 1u << 31; data = ptr[i];
    for (uint32_t b = 0; b < 32; b++) { if (CRC32 & 0x80000000) { CRC32 <<= 1; CRC32 ^= poly; } else CRC32 <<= 1; if (data & xbit) CRC32 ^= poly; xbit >>= 1; } }
  return CRC32;
}
template <typename MSG> class ChannelSubscriber {
 public:
  explicit ChannelSubscriber(const std::string&) {}
  void InitChannel(const std::function<void(const void*)>& handler, int64_t = 0) {
    if constexpr (std::is_same<MSG, unitree_hg::msg::dds_::LowCmd_>::value) {
      if (std::getenv("STUB_FOREIGN")) std::thread([handler] { while (true) { handler(nullptr); std::this_thread::sleep_for(std::chrono::milliseconds(10)); } }).detach();   // another publisher on rt/lowcmd
    }
    if constexpr (std::is_same<MSG, unitree_hg::msg::dds_::LowState_>::value) {
      std::thread([handler] {
        auto* msg = new unitree_hg::msg::dds_::LowState_();          // the same object every time, so padding bytes match the checksum
        uint32_t tick = 0;
        auto next = std::chrono::steady_clock::now();
        while (true) {
          next += std::chrono::microseconds(1000);
          msg->tick() = tick++;
          msg->mode_machine() = 6;
          float dq3 = 0.0f;                                    // STUB_DQ3_FILE: while the file exists, joint 3 reports the speed written in it (rad/s)
          if (const char* f3 = std::getenv("STUB_DQ3_FILE")) { FILE* fp = std::fopen(f3, "r"); if (fp) { double v = 0; if (std::fscanf(fp, "%lf", &v) == 1) dq3 = (float)v; std::fclose(fp); } }
          float ramp = std::getenv("STUB_RAMP") ? 1e-3f * (float)(tick % 100) : 0.0f;   // all joints move together: lets a test detect torn reads
          for (int i = 0; i < 29; ++i) { msg->motor_state().at(i).q() = 0.1f * std::sin(0.3f * i) + ramp; msg->motor_state().at(i).dq() = (i == 3 && dq3 != 0.0f) ? dq3 : 0.0f; msg->motor_state().at(i).tau_est() = 0.1f; msg->motor_state().at(i).temperature().at(0) = 40; }
          msg->imu_state().quaternion() = {1.0f, 0.0f, 0.0f, 0.0f};
          msg->crc() = stub_crc((uint32_t*)msg, (sizeof(unitree_hg::msg::dds_::LowState_) >> 2) - 1);
          const char* pause = std::getenv("STUB_PAUSE_FILE");
          if (!(pause && ::access(pause, F_OK) == 0)) handler(msg);   // pausing = the state stops arriving
          std::this_thread::sleep_until(next);
        }
      }).detach();
    }
  }
};
template <typename MSG> using ChannelSubscriberPtr = std::shared_ptr<ChannelSubscriber<MSG>>;
}}
