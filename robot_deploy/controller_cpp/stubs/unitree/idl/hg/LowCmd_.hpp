// LAPTOP STAND-IN for the Unitree hg message types (same accessor names as the real headers). Used only to compile and exercise the probe off the robot.
#pragma once
#include <array>
#include <cstdint>
namespace unitree_hg { namespace msg { namespace dds_ {
class MotorCmd_ {
  uint8_t mode_ = 0; float q_ = 0, dq_ = 0, tau_ = 0, kp_ = 0, kd_ = 0; uint32_t reserve_ = 0;
 public:
  uint8_t mode() const { return mode_; } uint8_t& mode() { return mode_; }
  float q() const { return q_; } float& q() { return q_; }
  float dq() const { return dq_; } float& dq() { return dq_; }
  float tau() const { return tau_; } float& tau() { return tau_; }
  float kp() const { return kp_; } float& kp() { return kp_; }
  float kd() const { return kd_; } float& kd() { return kd_; }
};
class LowCmd_ {
  uint8_t mode_pr_ = 0, mode_machine_ = 0; std::array<MotorCmd_, 35> motor_cmd_ = {}; std::array<uint32_t, 4> reserve_ = {}; uint32_t crc_ = 0;
 public:
  uint8_t mode_pr() const { return mode_pr_; } uint8_t& mode_pr() { return mode_pr_; }
  uint8_t mode_machine() const { return mode_machine_; } uint8_t& mode_machine() { return mode_machine_; }
  const std::array<MotorCmd_, 35>& motor_cmd() const { return motor_cmd_; } std::array<MotorCmd_, 35>& motor_cmd() { return motor_cmd_; }
  uint32_t crc() const { return crc_; } uint32_t& crc() { return crc_; }
};
}}}
