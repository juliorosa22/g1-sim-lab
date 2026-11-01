// C++ counterpart of `controller/run.py --stop-test`, as a PROBE: does the C++ SDK publish path avoid the rare 30-45 ms stalls seen from Python?
//
//   stoptest_probe [--net eth0] [--plan] [--publish --stage LABEL] [--hold-s 4] [--fade-s 0.5] [--damp-s 3] [--out DIR]
//
// Default is a DRY RUN: reads rt/lowstate, builds the commands it WOULD send (incl. the message fill and the checksum, so those costs are measured) and sends nothing.
// Publishing needs --publish AND --stage LABEL AND the environment variable CONTROLLER_CONFIRM=yes-LABEL, set only after the user's yes for this exact run (CLAUDE.md rule 2).
//
// Sequence (same as the Python stop test): HOLD the measured pose for --hold-s (gains 30% -> 100% over 2 s), then STOP: kp fades to 0 over --fade-s while the target follows the
// measured pose, then kd-only damping for --damp-s, then publishing ends (the motors then fall back to mode 0, limp). Robot hoisted, in debug mode, e-stop in hand.
// Safety: stale state (> 50 ms), NaN/inf, tilt > 25 deg, joint speed > 5 rad/s, |tau_est| above a per-motor cap, tracking error > 0.30 rad, motor > 70 C, STOP file, SIGINT/SIGTERM/SIGHUP
// all start the stop sequence. NOT included (on purpose, it is what we measure): the loop-period trip. Per tick it logs how late the loop woke up and how long the fill, the checksum and the
// DDS Write took, plus the state age.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelSubscriber;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;

namespace {
using Clock = std::chrono::steady_clock;
constexpr int N = 29;
constexpr double DT = 0.02;
inline bool absent(int i) { return i == 13 || i == 14; }  // waist roll / pitch: locked on this unit
double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

inline uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {  // as in Unitree's low-level examples
  uint32_t xbit = 0, data = 0, CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1u << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) { CRC32 <<= 1; CRC32 ^= dwPolynomial; } else CRC32 <<= 1;
      if (data & xbit) CRC32 ^= dwPolynomial;
      xbit >>= 1;
    }
  }
  return CRC32;
}

// ---- gains and caps: the v11 contract (configs/g1_v11_stand_turn.json) and controller/safety.py
struct Table { double kp[N], kd[N], cap[N]; };
Table make_table() {
  Table t{};
  const double lkp[6] = {100, 100, 100, 150, 60, 60}, lkd[6] = {4, 4, 4, 5, 3, 3}, lcap[6] = {60, 80, 60, 80, 40, 40};
  for (int side : {0, 6})
    for (int j = 0; j < 6; ++j) { t.kp[side + j] = lkp[j]; t.kd[side + j] = lkd[j]; t.cap[side + j] = lcap[j]; }
  t.kp[12] = 200; t.kd[12] = 6; t.cap[12] = 50;
  t.kp[13] = t.kp[14] = t.kd[13] = t.kd[14] = 0; t.cap[13] = t.cap[14] = 1e9;
  for (int i = 15; i < N; ++i) {
    bool wrist = (i >= 19 && i <= 21) || i >= 26;
    t.kp[i] = wrist ? 40 : 60; t.kd[i] = 3; t.cap[i] = 25;
  }
  return t;
}

// ---- state capture (DDS callback only converts and stores)
struct State {
  bool valid = false;
  double t = 0.0;
  uint8_t mm = 0;
  double q[N] = {}, dq[N] = {}, tau[N] = {}, temp[N] = {}, gyro[3] = {}, quat[4] = {1, 0, 0, 0};
};
std::mutex g_mtx;
State g_state;
std::atomic<uint64_t> g_n_state{0}, g_n_crc_bad{0}, g_n_foreign{0};
std::atomic<bool> g_ignore_state_crc{false};
std::atomic<int> g_signal{0};

void on_state(const void* msg) {
  const LowState* s = static_cast<const LowState*>(msg);
  g_n_state++;
  if (Crc32Core((uint32_t*)s, (sizeof(LowState) >> 2) - 1) != s->crc()) {
    g_n_crc_bad++;
    if (!g_ignore_state_crc) return;
  }
  State st;
  st.valid = true; st.t = now_s(); st.mm = s->mode_machine();
  for (int i = 0; i < N; ++i) {
    const auto& m = s->motor_state().at(i);
    st.q[i] = m.q(); st.dq[i] = m.dq(); st.tau[i] = m.tau_est(); st.temp[i] = m.temperature().at(0);
  }
  for (int k = 0; k < 3; ++k) st.gyro[k] = s->imu_state().gyroscope().at(k);
  for (int k = 0; k < 4; ++k) st.quat[k] = s->imu_state().quaternion().at(k);
  std::lock_guard<std::mutex> lk(g_mtx);
  g_state = st;
}
void on_cmd(const void*) { g_n_foreign++; }
State latest() { std::lock_guard<std::mutex> lk(g_mtx); return g_state; }
void on_signal(int s) { g_signal = s; }
// leave without running static destructors while SDK threads may still call back into them
[[noreturn]] void finish(int rc) { std::fflush(nullptr); std::_Exit(rc); }

double tilt_deg(const double* q) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double n = std::sqrt(w * w + x * x + y * y + z * z);
  if (n < 1e-9) return 180.0;
  x /= n; y /= n;
  double gz = -(1.0 - 2.0 * (x * x + y * y));
  return std::acos(std::max(-1.0, std::min(1.0, -gz))) * 180.0 / M_PI;
}
bool finite_all(const double* a, int n) { for (int i = 0; i < n; ++i) if (!std::isfinite(a[i])) return false; return true; }

// returns "" when fine, else the reason
std::string check(const State& s, double now, const Table& tb, const double* prev_target, bool enforce_tracking) {
  char buf[160];
  if (!s.valid) return "no state yet";
  if (now - s.t > 0.05) { std::snprintf(buf, sizeof buf, "stale state (%.0f ms old)", (now - s.t) * 1e3); return buf; }
  if (!finite_all(s.q, N) || !finite_all(s.dq, N) || !finite_all(s.tau, N) || !finite_all(s.gyro, 3) || !finite_all(s.quat, 4)) return "NaN/inf in state";
  double tl = tilt_deg(s.quat);
  if (tl > 25.0) { std::snprintf(buf, sizeof buf, "tilt %.1f deg > 25", tl); return buf; }
  for (int i = 0; i < N; ++i) {
    if (absent(i)) continue;
    if (std::fabs(s.dq[i]) > 5.0) { std::snprintf(buf, sizeof buf, "joint %d speed %.1f rad/s > 5", i, s.dq[i]); return buf; }
    if (std::fabs(s.tau[i]) > tb.cap[i]) { std::snprintf(buf, sizeof buf, "motor %d |tau_est| %.1f N*m > cap %.0f", i, std::fabs(s.tau[i]), tb.cap[i]); return buf; }
    if (s.temp[i] > 70.0) { std::snprintf(buf, sizeof buf, "motor %d at %d C", i, (int)s.temp[i]); return buf; }
    if (enforce_tracking && std::fabs(s.q[i] - prev_target[i]) > 0.30) { std::snprintf(buf, sizeof buf, "joint %d tracking error %.2f rad > 0.30", i, s.q[i] - prev_target[i]); return buf; }
  }
  return "";
}

struct Stats { double mean = 0, p50 = 0, p99 = 0, p999 = 0, mx = 0; int over5 = 0, over15 = 0; };
Stats stats(std::vector<double> v) {
  Stats s;
  if (v.empty()) return s;
  double sum = 0; for (double x : v) { sum += x; if (x > 5.0) s.over5++; if (x > 15.0) s.over15++; }
  std::sort(v.begin(), v.end());
  auto pct = [&](double p) { return v[std::min(v.size() - 1, (size_t)(p / 100.0 * (v.size() - 1) + 0.5))]; };
  s.mean = sum / v.size(); s.p50 = pct(50); s.p99 = pct(99); s.p999 = pct(99.9); s.mx = v.back();
  return s;
}
void print_stats(const char* name, const std::vector<double>& v) {
  Stats s = stats(v);
  std::printf("  %-14s mean %7.3f  p50 %7.3f  p99 %7.3f  p99.9 %7.3f  max %7.3f ms   (> 5 ms: %d, > 15 ms: %d)\n", name, s.mean, s.p50, s.p99, s.p999, s.mx, s.over5, s.over15);
}
bool file_exists(const std::string& p) { struct stat sb; return stat(p.c_str(), &sb) == 0; }
}  // namespace

int main(int argc, char** argv) {
  std::string net = "eth0", stage, outdir;
  bool plan = false, publish = false;
  double hold_s = 4.0, fade_s = 0.5, damp_s = 3.0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
    if (a == "--net") net = next();
    else if (a == "--plan") plan = true;
    else if (a == "--publish") publish = true;
    else if (a == "--stage") stage = next();
    else if (a == "--hold-s") hold_s = std::stod(next());
    else if (a == "--fade-s") fade_s = std::stod(next());
    else if (a == "--damp-s") damp_s = std::stod(next());
    else if (a == "--out") outdir = next();
    else if (a == "--ignore-state-crc") g_ignore_state_crc = true;
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); std::exit(2); }
  }
  if (publish) {
    const char* c = std::getenv("CONTROLLER_CONFIRM");
    if (stage.empty() || !c || std::string(c) != "yes-" + stage) {
      std::fprintf(stderr, "REFUSED: publishing needs --stage LABEL and CONTROLLER_CONFIRM=yes-LABEL, set only after the user's yes for this exact run\n");
      finish(2);
    }
  }
  const Table tb = make_table();

  ChannelFactory::Instance()->Init(0, net);
  std::shared_ptr<ChannelSubscriber<LowState>> sub_state(new ChannelSubscriber<LowState>("rt/lowstate"));
  sub_state->InitChannel(on_state, 1);
  std::shared_ptr<ChannelSubscriber<LowCmd>> sub_cmd(new ChannelSubscriber<LowCmd>("rt/lowcmd"));
  sub_cmd->InitChannel(on_cmd, 1);

  // ---- preflight (read-only)
  std::vector<std::string> problems;
  double t_wait = now_s();
  while (!latest().valid && now_s() - t_wait < 2.0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  if (!latest().valid) problems.push_back("no valid rt/lowstate within 2 s (state CRC mismatches so far: " + std::to_string(g_n_crc_bad.load()) + " of " + std::to_string(g_n_state.load()) + ")");
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));  // also listens for another rt/lowcmd publisher
  State s0 = latest();
  if (s0.valid) {
    double mx_dq = 0, mx_t = 0;
    for (int i = 0; i < N; ++i) if (!absent(i)) { mx_dq = std::max(mx_dq, std::fabs(s0.dq[i])); mx_t = std::max(mx_t, s0.temp[i]); }
    if (publish && g_n_foreign > 0) problems.push_back("another publisher is active on rt/lowcmd (" + std::to_string(g_n_foreign.load()) + " msgs): not in debug mode");
    if (mx_dq > 0.5) problems.push_back("robot is moving: max |dq| " + std::to_string(mx_dq));
    if (mx_t > 60.0) problems.push_back("a motor is at " + std::to_string((int)mx_t) + " C");
  }
  std::printf("PLAN stoptest_probe (C++) | %s | 50 Hz | HOLD %.1f s (gains 30%%->100%% over 2 s) then STOP: kp fade %.1f s, kd-only damping %.1f s, then publishing ends\n",
              publish ? "PUBLISHING" : "DRY RUN (nothing is sent)", hold_s, fade_s, damp_s);
  std::printf("  gains legs kp 100/100/100/150/60/60 kd 4/4/4/5/3/3 | waist yaw 200/6 | arms 60/3, wrists 40/3 | waist roll/pitch 0/0 | target = the measured pose\n");
  std::printf("  safety: stale > 50 ms | NaN | tilt > 25 deg | speed > 5 rad/s | |tau_est| caps (hip 60/80/60, knee 80, ankle 40, waist 50, arms 25) | tracking > 0.30 rad | motor > 70 C | STOP file | signals. NO loop-period trip (it is measured).\n");
  std::printf("PREFLIGHT states received %llu (CRC mismatches %llu), foreign lowcmd msgs %llu\n", (unsigned long long)g_n_state.load(), (unsigned long long)g_n_crc_bad.load(), (unsigned long long)g_n_foreign.load());
  std::printf("PREFLIGHT problems:");
  if (problems.empty()) std::printf(" none"); else for (auto& p : problems) std::printf(" [%s]", p.c_str());
  std::printf("\n");
  if (plan) { std::printf("PLAN ONLY: nothing published\n"); finish(problems.empty() || !publish ? 0 : 1); }
  if (publish && !problems.empty()) { std::printf("NOT RUNNING: fix the problems above\n"); finish(1); }
  if (!s0.valid) { std::printf("NOT RUNNING: no state\n"); finish(1); }

  std::shared_ptr<ChannelPublisher<LowCmd>> pub;
  if (publish) { pub.reset(new ChannelPublisher<LowCmd>("rt/lowcmd")); pub->InitChannel(); }
  std::signal(SIGINT, on_signal); std::signal(SIGTERM, on_signal); std::signal(SIGHUP, on_signal);
  if (!outdir.empty()) { std::string cmd = "mkdir -p '" + outdir + "'"; if (std::system(cmd.c_str()) != 0) std::fprintf(stderr, "could not create %s\n", outdir.c_str()); }

  // ---- control loop
  LowCmd cmd;
  double q_start[N], target[N], prev_target[N];
  for (int i = 0; i < N; ++i) q_start[i] = absent(i) ? 0.0 : s0.q[i];
  std::copy(q_start, q_start + N, prev_target);
  std::vector<std::string> rows;
  std::vector<double> v_write, v_fill, v_crc, v_send, v_wake, v_period, v_cycle, v_age;
  std::string stop_reason;
  bool stopping = false, done = false;
  double t_stop = 0.0, stop_scale = 1.0, scale_now = 1.0, last_wake = -1.0;
  char head[] = "tick,t,phase,wake_late_ms,period_ms,cycle_ms,state_age_ms,fill_ms,crc_ms,write_ms,send_total_ms,max_abs_dq,max_abs_tau,tilt_deg,kp_knee,q_knee,trip\n";
  rows.push_back(head);
  auto next_t = Clock::now();
  const double t0 = now_s();
  int tick = 0;
  uint64_t n_written_ok = 0, n_written_fail = 0;
  while (!done) {
    next_t += std::chrono::microseconds((int64_t)(DT * 1e6));
    std::this_thread::sleep_until(next_t);
    double t_wake = now_s();
    double wake_late = (std::chrono::duration<double>(Clock::now() - next_t).count()) * 1e3;
    double period_ms = last_wake < 0 ? DT * 1e3 : (t_wake - last_wake) * 1e3;
    last_wake = t_wake;
    double t = t_wake - t0;
    State s = latest();
    std::string trip;
    if (!stopping) {
      if (g_signal) trip = "signal";
      else if (!outdir.empty() && file_exists(outdir + "/STOP")) trip = "STOP file";
      else trip = check(s, t_wake, tb, prev_target, publish && tick > 5);
      if (!trip.empty()) { stopping = true; t_stop = t; stop_reason = trip; stop_scale = scale_now; }
      else if (t >= hold_s) { stopping = true; t_stop = t; stop_reason = "finished (stop test)"; stop_scale = scale_now; }
    }
    double kp[N], kd[N];
    int phase = 1;  // HOLD
    if (!stopping) {
      double scale = std::max(0.3, std::min(1.0, t / 2.0));
      scale_now = scale;
      for (int i = 0; i < N; ++i) { target[i] = q_start[i]; kp[i] = tb.kp[i] * scale; kd[i] = tb.kd[i] * scale; }
    } else {
      phase = 4;
      double u = t - t_stop;
      if (u > fade_s + damp_s) { done = true; break; }
      bool ok = s.valid && finite_all(s.q, N);
      for (int i = 0; i < N; ++i) {
        target[i] = ok && !absent(i) ? s.q[i] : prev_target[i];  // follow the measured pose: no jump while kp fades
        kp[i] = tb.kp[i] * stop_scale * std::max(0.0, 1.0 - u / fade_s);
        kd[i] = tb.kd[i];
      }
    }
    for (int i = 0; i < N; ++i) if (absent(i)) { target[i] = 0.0; kp[i] = 0.0; kd[i] = 0.0; }
    std::copy(target, target + N, prev_target);

    // ---- fill, checksum, write: each timed
    double a0 = now_s();
    cmd.mode_pr() = 0;
    cmd.mode_machine() = s.mm;
    for (int i = 0; i < N; ++i) {
      auto& m = cmd.motor_cmd().at(i);
      m.mode() = 1; m.q() = (float)target[i]; m.dq() = 0.0f; m.tau() = 0.0f; m.kp() = (float)kp[i]; m.kd() = (float)kd[i];
    }
    double a1 = now_s();
    cmd.crc() = Crc32Core((uint32_t*)&cmd, (sizeof(cmd) >> 2) - 1);
    double a2 = now_s();
    bool wrote = true;
    if (publish) { wrote = pub->Write(cmd); if (wrote) n_written_ok++; else n_written_fail++; }
    double a3 = now_s();
    double fill_ms = (a1 - a0) * 1e3, crc_ms = (a2 - a1) * 1e3, write_ms = (a3 - a2) * 1e3;
    double cycle_ms = (a3 - t_wake) * 1e3;
    double mxdq = 0, mxt = 0;
    for (int i = 0; i < N; ++i) if (!absent(i)) { mxdq = std::max(mxdq, std::fabs(s.dq[i])); mxt = std::max(mxt, std::fabs(s.tau[i])); }
    v_write.push_back(write_ms); v_fill.push_back(fill_ms); v_crc.push_back(crc_ms); v_send.push_back(fill_ms + crc_ms + write_ms);
    v_wake.push_back(wake_late); v_period.push_back(period_ms); v_cycle.push_back(cycle_ms); v_age.push_back(s.valid ? (t_wake - s.t) * 1e3 : -1.0);
    char row[512];
    std::snprintf(row, sizeof row, "%d,%.4f,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.1f,%.5f,%s\n", tick, t, phase, wake_late, period_ms, cycle_ms,
                  v_age.back(), fill_ms, crc_ms, write_ms, fill_ms + crc_ms + write_ms, mxdq, mxt, s.valid ? tilt_deg(s.quat) : -1.0, kp[3], s.q[3], trip.c_str());
    rows.emplace_back(row);
    tick++;
  }
  // ---- write the log and summarise
  if (!outdir.empty()) { std::ofstream f(outdir + "/log.csv"); for (auto& r : rows) f << r; }
  std::printf("END reason: %s | ticks %d | writes ok %llu failed %llu | state msgs %llu (CRC mismatches %llu)\n", stop_reason.empty() ? "ended" : stop_reason.c_str(), tick,
              (unsigned long long)n_written_ok, (unsigned long long)n_written_fail, (unsigned long long)g_n_state.load(), (unsigned long long)g_n_crc_bad.load());
  std::printf("TIMING (ms)\n");
  print_stats("write (DDS)", v_write); print_stats("fill", v_fill); print_stats("checksum", v_crc); print_stats("send total", v_send);
  print_stats("wake lateness", v_wake); print_stats("tick period", v_period); print_stats("tick cycle", v_cycle); print_stats("state age", v_age);
  finish(0);
}
