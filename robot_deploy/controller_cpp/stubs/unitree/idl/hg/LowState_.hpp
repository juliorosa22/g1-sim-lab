// LAPTOP STAND-IN (see LowCmd_.hpp)
#pragma once
#include <array>
#include <cstdint>
namespace unitree_hg { namespace msg { namespace dds_ {
class IMUState_ {
  std::array<float, 4> quaternion_ = {1, 0, 0, 0}; std::array<float, 3> gyroscope_ = {}, accelerometer_ = {}, rpy_ = {}; int16_t temperature_ = 0;
 public:
  const std::array<float, 4>& quaternion() const { return quaternion_; } std::array<float, 4>& quaternion() { return quaternion_; }
  const std::array<float, 3>& gyroscope() const { return gyroscope_; } std::array<float, 3>& gyroscope() { return gyroscope_; }
};
class MotorState_ {
  uint8_t mode_ = 0; float q_ = 0, dq_ = 0, ddq_ = 0, tau_est_ = 0; std::array<int16_t, 2> temperature_ = {}; float vol_ = 0;
 public:
  uint8_t mode() const { return mode_; } uint8_t& mode() { return mode_; }
  float q() const { return q_; } float& q() { return q_; }
  float dq() const { return dq_; } float& dq() { return dq_; }
  float tau_est() const { return tau_est_; } float& tau_est() { return tau_est_; }
  const std::array<int16_t, 2>& temperature() const { return temperature_; } std::array<int16_t, 2>& temperature() { return temperature_; }
};
class LowState_ {
  uint32_t version_[2] = {}; uint8_t mode_pr_ = 0, mode_machine_ = 0; uint32_t tick_ = 0; IMUState_ imu_state_; std::array<MotorState_, 35> motor_state_ = {};
  std::array<uint8_t, 40> wireless_remote_ = {}; std::array<uint32_t, 4> reserve_ = {}; uint32_t crc_ = 0;
 public:
  uint8_t mode_machine() const { return mode_machine_; } uint8_t& mode_machine() { return mode_machine_; }
  uint32_t tick() const { return tick_; } uint32_t& tick() { return tick_; }
  const IMUState_& imu_state() const { return imu_state_; } IMUState_& imu_state() { return imu_state_; }
  const std::array<MotorState_, 35>& motor_state() const { return motor_state_; } std::array<MotorState_, 35>& motor_state() { return motor_state_; }
  uint32_t crc() const { return crc_; } uint32_t& crc() { return crc_; }
};
}}}
