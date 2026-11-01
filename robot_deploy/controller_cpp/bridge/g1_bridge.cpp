// g1_bridge: owns ALL DDS traffic for the G1 controller. The controller (Python today) reads robot state and writes commands through shared memory (see DESIGN.md).
//
//   g1_bridge [--net eth0] [--shm /dev/shm/g1_bridge] [--publish --stage LABEL] [--dead-ms 60] [--fade-s 0.5] [--damp-s 3] [--linger-s 3] [--out DIR] [--on-arrival] [--max-err 0.35] [--max-speed 5]
//   g1_bridge --print-layout | --print-tables        (no DDS, no shared memory: used by the tests)
//
// Without --publish it runs in SHADOW mode: the whole state machine, firewall and stop sequence run, but no publisher exists, so nothing can reach the robot.
// Publishing needs --publish AND --stage LABEL AND CONTROLLER_CONFIRM=yes-LABEL in the environment, set only after the user's yes for this exact run (CLAUDE.md rule 2).
// One run per process: after STOPPED / REFUSED it lingers --linger-s and exits.
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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "shm_protocol.h"

using namespace g1bridge;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelSubscriber;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;

namespace {
using Clock = std::chrono::steady_clock;
constexpr double DT = 0.02;
inline bool absent(int i) { return i == 13 || i == 14; }
double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
[[noreturn]] void finish(int rc) { std::fflush(nullptr); std::_Exit(rc); }

inline uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0, data = 0, CRC32 = 0xFFFFFFFF;
  const uint32_t poly = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1u << 31; data = ptr[i];
    for (uint32_t b = 0; b < 32; b++) {
      if (CRC32 & 0x80000000) { CRC32 <<= 1; CRC32 ^= poly; } else CRC32 <<= 1;
      if (data & xbit) CRC32 ^= poly;
      xbit >>= 1;
    }
  }
  return CRC32;
}

// contract values (configs/g1_v11_stand_turn.json) and the controller's torque caps (controller/safety.py); tests compare --print-tables with them
struct Table { double kp[N], kd[N], cap[N]; };
Table make_table() {
  Table t{};
  const double lkp[6] = {100, 100, 100, 150, 60, 60}, lkd[6] = {4, 4, 4, 5, 3, 3}, lcap[6] = {60, 80, 60, 80, 40, 40};
  for (int side : {0, 6})
    for (int j = 0; j < 6; ++j) { t.kp[side + j] = lkp[j]; t.kd[side + j] = lkd[j]; t.cap[side + j] = lcap[j]; }
  t.kp[12] = 200; t.kd[12] = 6; t.cap[12] = 50;
  t.cap[13] = t.cap[14] = 1e9;
  for (int i = 15; i < N; ++i) { bool wrist = (i >= 19 && i <= 21) || i >= 26; t.kp[i] = wrist ? 40 : 60; t.kd[i] = 3; t.cap[i] = 25; }
  return t;
}

struct Shm { g1bridge::Header* h; StateBlock* st; CmdBlock* cmd; StatusBlock* stat; int fd; };
Shm g_shm{};

struct LocalState {
  bool valid = false;
  double t = 0.0;
  uint8_t mm = 0;
  double q[N] = {}, dq[N] = {}, tau[N] = {}, temp[N] = {}, gyro[3] = {}, quat[4] = {1, 0, 0, 0};
};
std::mutex g_mtx;
LocalState g_ls;
std::atomic<uint64_t> g_n_msgs{0}, g_n_crc_bad{0}, g_n_foreign{0};
std::atomic<bool> g_ignore_state_crc{false};
double g_max_speed = 5.0;   // state check: any joint faster than this (rad/s) stops the run (--max-speed)
double g_max_err = 0.35;   // firewall: |target - measured q| allowed per joint (--max-err); the FIRST command is always limited to 0.10
std::atomic<int> g_signal{0};

void on_signal(int s) { g_signal = s; }
void on_cmd(const void*) { g_n_foreign++; }
void on_state(const void* msg) {
  const LowState* s = static_cast<const LowState*>(msg);
  g_n_msgs++;
  if (Crc32Core((uint32_t*)s, (sizeof(LowState) >> 2) - 1) != s->crc()) {
    g_n_crc_bad++;
    if (!g_ignore_state_crc) return;
  }
  double t = now_s();
  StatePayload sp;
  std::memset(&sp, 0, sizeof sp);
  sp.t_arrival = t; sp.n_msgs = g_n_msgs; sp.n_crc_bad = g_n_crc_bad; sp.n_foreign_cmd = g_n_foreign;
  sp.tick = s->tick(); sp.mode_machine = s->mode_machine();
  LocalState ls;
  ls.valid = true; ls.t = t; ls.mm = s->mode_machine();
  for (int i = 0; i < N; ++i) {
    const auto& m = s->motor_state().at(i);
    sp.q[i] = ls.q[i] = m.q(); sp.dq[i] = ls.dq[i] = m.dq(); sp.tau[i] = ls.tau[i] = m.tau_est();
    sp.temp[i] = ls.temp[i] = m.temperature().at(0); sp.mode[i] = m.mode();
  }
  for (int k = 0; k < 3; ++k) sp.gyro[k] = ls.gyro[k] = s->imu_state().gyroscope().at(k);
  for (int k = 0; k < 4; ++k) sp.quat[k] = ls.quat[k] = s->imu_state().quaternion().at(k);
  write_block(*g_shm.st, sp);
  std::lock_guard<std::mutex> lk(g_mtx);
  g_ls = ls;
}
LocalState latest() { std::lock_guard<std::mutex> lk(g_mtx); return g_ls; }

double tilt_deg(const double* q) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double n = std::sqrt(w * w + x * x + y * y + z * z);
  if (n < 1e-9) return 180.0;
  x /= n; y /= n;
  double gz = -(1.0 - 2.0 * (x * x + y * y));
  (void)w; (void)z;
  return std::acos(std::max(-1.0, std::min(1.0, -gz))) * 180.0 / M_PI;
}
bool finite_all(const double* a, int n) { for (int i = 0; i < n; ++i) if (!std::isfinite(a[i])) return false; return true; }

// state checks: a second, independent copy of the controller's safety limits. Returns "" or the reason.
std::string check_state(const LocalState& s, double now, const Table& tb) {
  char buf[160];
  if (!s.valid) return "no state";
  if (now - s.t > 0.05) { std::snprintf(buf, sizeof buf, "stale state (%.0f ms old)", (now - s.t) * 1e3); return buf; }
  if (!finite_all(s.q, N) || !finite_all(s.dq, N) || !finite_all(s.tau, N) || !finite_all(s.gyro, 3) || !finite_all(s.quat, 4)) return "NaN/inf in state";
  double tl = tilt_deg(s.quat);
  if (tl > 25.0) { std::snprintf(buf, sizeof buf, "tilt %.1f deg > 25", tl); return buf; }
  for (int i = 0; i < N; ++i) {
    if (absent(i)) continue;
    if (std::fabs(s.dq[i]) > g_max_speed) { std::snprintf(buf, sizeof buf, "joint %d speed %.1f rad/s > %g", i, s.dq[i], g_max_speed); return buf; }
    if (std::fabs(s.tau[i]) > tb.cap[i]) { std::snprintf(buf, sizeof buf, "motor %d |tau_est| %.1f N*m > cap %.0f", i, std::fabs(s.tau[i]), tb.cap[i]); return buf; }
    if (s.temp[i] > 70.0) { std::snprintf(buf, sizeof buf, "motor %d at %d C", i, (int)s.temp[i]); return buf; }
  }
  return "";
}

// the firewall: gross-error catch on every command. first = the enabling command (must start where the robot is)
std::string firewall(const CmdPayload& c, const LocalState& s, const double* prev_target, bool first, const Table& tb) {
  char buf[160];
  for (int i = 0; i < N; ++i)
    if (!std::isfinite(c.target[i]) || !std::isfinite(c.kp[i]) || !std::isfinite(c.kd[i])) return "non-finite command";
  for (int i = 0; i < N; ++i) {
    if (absent(i)) { if (c.kp[i] != 0.0 || c.kd[i] != 0.0) return "waist roll/pitch must have kp = kd = 0"; continue; }
    if (c.kp[i] < 0.0 || c.kp[i] > tb.kp[i] * 1.001) { std::snprintf(buf, sizeof buf, "motor %d kp %.1f outside [0, %.0f]", i, c.kp[i], tb.kp[i]); return buf; }
    if (c.kd[i] < 0.0 || c.kd[i] > tb.kd[i] * 1.001) { std::snprintf(buf, sizeof buf, "motor %d kd %.2f outside [0, %.1f]", i, c.kd[i], tb.kd[i]); return buf; }
    if (s.valid) {
      double lim = first ? 0.10 : g_max_err;
      if (std::fabs(c.target[i] - s.q[i]) > lim) { std::snprintf(buf, sizeof buf, "motor %d target %.2f rad from the measured pose (limit %.2f)", i, c.target[i] - s.q[i], lim); return buf; }
    }
    if (!first && std::fabs(c.target[i] - prev_target[i]) > 0.30) { std::snprintf(buf, sizeof buf, "motor %d target step %.2f rad in one tick (limit 0.30)", i, c.target[i] - prev_target[i]); return buf; }
  }
  return "";
}

Shm open_shm(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0600);
  if (fd < 0) { std::perror("open shm"); finish(2); }
  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) { std::fprintf(stderr, "REFUSED: another g1_bridge holds %s\n", path.c_str()); finish(2); }
  if (::ftruncate(fd, 0) != 0 || ::ftruncate(fd, (off_t)TOTAL) != 0) { std::perror("ftruncate"); finish(2); }  // zero-filled
  void* m = ::mmap(nullptr, TOTAL, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (m == MAP_FAILED) { std::perror("mmap"); finish(2); }
  Shm s{};
  s.fd = fd;
  s.h = static_cast<g1bridge::Header*>(m);
  s.h->magic = MAGIC; s.h->version = VERSION; s.h->n_motor = N; s.h->header_size = sizeof(g1bridge::Header);
  s.h->off_state = OFF_STATE; s.h->off_cmd = OFF_CMD; s.h->off_status = OFF_STATUS; s.h->total_size = TOTAL;
  s.st = reinterpret_cast<StateBlock*>(static_cast<char*>(m) + OFF_STATE);
  s.cmd = reinterpret_cast<CmdBlock*>(static_cast<char*>(m) + OFF_CMD);
  s.stat = reinterpret_cast<StatusBlock*>(static_cast<char*>(m) + OFF_STATUS);
  return s;
}

struct Stats { double mean = 0, p50 = 0, p99 = 0, mx = 0; };
Stats stats(std::vector<double> v) {
  Stats s;
  if (v.empty()) return s;
  double sum = 0; for (double x : v) sum += x;
  std::sort(v.begin(), v.end());
  s.mean = sum / v.size(); s.p50 = v[v.size() / 2]; s.p99 = v[std::min(v.size() - 1, (size_t)(0.99 * (v.size() - 1) + 0.5))]; s.mx = v.back();
  return s;
}
}  // namespace

#define OFFS(T, m) (long)offsetof(T, m)
void print_layout() {
  std::printf("{\"header\":{\"size\":%zu},\"total\":%zu,\"off\":{\"state\":%zu,\"cmd\":%zu,\"status\":%zu},", sizeof(g1bridge::Header), TOTAL, OFF_STATE, OFF_CMD, OFF_STATUS);
  std::printf("\"state\":{\"size\":%zu,\"seq\":%ld,\"chk_xor\":%ld,\"chk_sum\":%ld,\"t_arrival\":%ld,\"n_msgs\":%ld,\"n_crc_bad\":%ld,\"n_foreign_cmd\":%ld,\"tick\":%ld,\"mode_machine\":%ld,\"q\":%ld,\"dq\":%ld,\"tau\":%ld,\"temp\":%ld,\"mode\":%ld,\"gyro\":%ld,\"quat\":%ld},",
              sizeof(StateBlock), OFFS(StateBlock, seq), OFFS(StateBlock, chk_xor), OFFS(StateBlock, chk_sum), OFFS(StateBlock, p.t_arrival), OFFS(StateBlock, p.n_msgs), OFFS(StateBlock, p.n_crc_bad),
              OFFS(StateBlock, p.n_foreign_cmd), OFFS(StateBlock, p.tick), OFFS(StateBlock, p.mode_machine), OFFS(StateBlock, p.q), OFFS(StateBlock, p.dq), OFFS(StateBlock, p.tau), OFFS(StateBlock, p.temp),
              OFFS(StateBlock, p.mode), OFFS(StateBlock, p.gyro), OFFS(StateBlock, p.quat));
  std::printf("\"cmd\":{\"size\":%zu,\"seq\":%ld,\"chk_xor\":%ld,\"chk_sum\":%ld,\"counter\":%ld,\"t_client\":%ld,\"flags\":%ld,\"target\":%ld,\"kp\":%ld,\"kd\":%ld},",
              sizeof(CmdBlock), OFFS(CmdBlock, seq), OFFS(CmdBlock, chk_xor), OFFS(CmdBlock, chk_sum), OFFS(CmdBlock, p.counter), OFFS(CmdBlock, p.t_client), OFFS(CmdBlock, p.flags), OFFS(CmdBlock, p.target),
              OFFS(CmdBlock, p.kp), OFFS(CmdBlock, p.kd));
  std::printf("\"status\":{\"size\":%zu,\"seq\":%ld,\"chk_xor\":%ld,\"chk_sum\":%ld,\"state\":%ld,\"reason_code\":%ld,\"reason\":%ld,\"n_published\":%ld,\"n_cmd_seen\":%ld,\"n_cmd_rejected\":%ld,\"n_deadman\":%ld,\"t_started\":%ld,\"t_heartbeat\":%ld,\"t_last_publish\":%ld,\"last_write_ms\":%ld,\"max_write_ms\":%ld,\"cmd_age_ms\":%ld,\"publishing\":%ld}}\n",
              sizeof(StatusBlock), OFFS(StatusBlock, seq), OFFS(StatusBlock, chk_xor), OFFS(StatusBlock, chk_sum), OFFS(StatusBlock, p.state), OFFS(StatusBlock, p.reason_code), OFFS(StatusBlock, p.reason),
              OFFS(StatusBlock, p.n_published), OFFS(StatusBlock, p.n_cmd_seen), OFFS(StatusBlock, p.n_cmd_rejected), OFFS(StatusBlock, p.n_deadman), OFFS(StatusBlock, p.t_started), OFFS(StatusBlock, p.t_heartbeat),
              OFFS(StatusBlock, p.t_last_publish), OFFS(StatusBlock, p.last_write_ms), OFFS(StatusBlock, p.max_write_ms), OFFS(StatusBlock, p.cmd_age_ms), OFFS(StatusBlock, p.publishing));
}
void print_tables() {
  Table t = make_table();
  std::printf("{\"kp\":[");
  for (int i = 0; i < N; ++i) std::printf("%s%g", i ? "," : "", t.kp[i]);
  std::printf("],\"kd\":[");
  for (int i = 0; i < N; ++i) std::printf("%s%g", i ? "," : "", t.kd[i]);
  std::printf("],\"cap\":[");
  for (int i = 0; i < N; ++i) std::printf("%s%g", i ? "," : "", t.cap[i] > 1e8 ? 1e9 : t.cap[i]);
  std::printf("]}\n");
}

int main(int argc, char** argv) {
  std::string net = "eth0", shm_path = "/dev/shm/g1_bridge", stage, outdir;
  bool publish = false;
  double dead_ms = 60.0, fade_s = 0.5, damp_s = 3.0, linger_s = 3.0;
  bool on_arrival = false;   // ACTIVE only: publish a new command as soon as it arrives instead of at the next fixed 50 Hz tick
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
    if (a == "--print-layout") { print_layout(); return 0; }
    else if (a == "--print-tables") { print_tables(); return 0; }
    else if (a == "--net") net = next();
    else if (a == "--shm") shm_path = next();
    else if (a == "--publish") publish = true;
    else if (a == "--stage") stage = next();
    else if (a == "--dead-ms") dead_ms = std::stod(next());
    else if (a == "--fade-s") fade_s = std::stod(next());
    else if (a == "--damp-s") damp_s = std::stod(next());
    else if (a == "--linger-s") linger_s = std::stod(next());
    else if (a == "--out") outdir = next();
    else if (a == "--on-arrival") on_arrival = true;
    else if (a == "--max-speed") { g_max_speed = std::stod(next()); if (!(g_max_speed >= 1.0 && g_max_speed <= 20.0)) { std::fprintf(stderr, "--max-speed must be between 1 and 20 rad/s\n"); std::exit(2); } }
    else if (a == "--max-err") { g_max_err = std::stod(next()); if (!(g_max_err >= 0.1 && g_max_err <= 1.0)) { std::fprintf(stderr, "--max-err must be between 0.1 and 1.0 rad\n"); std::exit(2); } }
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
  g_shm = open_shm(shm_path);
  const double t_start = now_s();

  ChannelFactory::Instance()->Init(0, net);
  std::shared_ptr<ChannelSubscriber<LowState>> sub_state(new ChannelSubscriber<LowState>("rt/lowstate"));
  sub_state->InitChannel(on_state, 1);
  std::shared_ptr<ChannelSubscriber<LowCmd>> sub_cmd(new ChannelSubscriber<LowCmd>("rt/lowcmd"));
  sub_cmd->InitChannel(on_cmd, 1);
  std::shared_ptr<ChannelPublisher<LowCmd>> pub;
  if (publish) { pub.reset(new ChannelPublisher<LowCmd>("rt/lowcmd")); pub->InitChannel(); }
  std::signal(SIGINT, on_signal); std::signal(SIGTERM, on_signal); std::signal(SIGHUP, on_signal);
  if (!outdir.empty()) { std::string c = "mkdir -p '" + outdir + "'"; if (std::system(c.c_str()) != 0) std::fprintf(stderr, "could not create %s\n", outdir.c_str()); }
  std::printf("g1_bridge %s | shm %s | dead-man %.0f ms | stop: fade %.1f s + damping %.1f s\n", publish ? "PUBLISHING" : "SHADOW (no publisher)", shm_path.c_str(), dead_ms, fade_s, damp_s);

  StatusPayload sp;
  std::memset(&sp, 0, sizeof sp);
  sp.t_started = t_start; sp.publishing = publish ? 1 : 0;
  auto set_status = [&](uint32_t state, uint32_t code, const std::string& reason) {
    sp.state = state; sp.reason_code = code;
    std::memset(sp.reason, 0, sizeof sp.reason);
    std::strncpy(sp.reason, reason.c_str(), sizeof sp.reason - 1);
  };
  set_status(ST_IDLE, 0, "waiting for a command with ENABLE");

  // ---- run state
  CmdPayload c, last_good;
  std::memset(&c, 0, sizeof c); std::memset(&last_good, 0, sizeof last_good);
  uint64_t last_counter = 0;
  double t_last_new = 0.0, t_stop = 0.0, t_end = 0.0;
  double prev_target[N] = {};
  int rejects = 0;
  uint32_t state = ST_IDLE;
  std::string reason;
  LowCmd cmd;
  std::vector<double> v_write, v_wake, v_period, v_cmdage;
  std::vector<std::string> rows;
  rows.push_back("tick,t,state,cmd_counter,cmd_age_ms,wake_late_ms,period_ms,state_age_ms,fill_ms,crc_ms,write_ms,write_to_pub_ms,state_to_pub_ms,rejected,reason\n");
  auto next_t = Clock::now();
  double last_wake = -1.0;
  bool prev_fresh = false; double prev_t = 0.0;   // the previous tick published a NEW command / when
  int tick = 0;
  uint64_t n_cmd_seen = 0;

  auto go_stopping = [&](const std::string& why, uint32_t code, double now) {
    if (state != ST_ACTIVE) return;
    state = ST_STOPPING; t_stop = now; reason = why;
    set_status(ST_STOPPING, code, why);
    std::printf("STOPPING: %s\n", why.c_str());
  };
  auto go_final = [&](uint32_t st, uint32_t code, const std::string& why, double now) {
    state = st; reason = why; t_end = now;
    set_status(st, code, why);
    std::printf("%s: %s\n", st == ST_REFUSED ? "REFUSED" : "STOPPED", why.c_str());
  };

  while (true) {
    next_t += std::chrono::microseconds((int64_t)(DT * 1e6));
    if (on_arrival && state == ST_ACTIVE) {
      // Wait for the schedule OR for a new command: publish it at once (the controller's phase decides, not ours). Two guards: a new command right after another new
      // one (< 10 ms, a burst) waits for the schedule, and when nothing arrives the schedule still fires (the stale-repeat and dead-man logic is unchanged).
      bool early = false;
      while (Clock::now() < next_t) {
        uint64_t sq = g_shm.cmd->seq.load(std::memory_order_acquire);
        if (!(sq & 1) && g_shm.cmd->p.counter != last_counter && !(prev_fresh && now_s() - prev_t < 0.010)) { early = true; break; }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
      if (early) next_t = Clock::now() + std::chrono::milliseconds(5);   // re-phase: the next scheduled tick (the stale-repeat deadline) is one period + 5 ms after this arrival, so a controller that is a little late is published on arrival, not after a repeat
    } else {
      std::this_thread::sleep_until(next_t);
    }
    double now = now_s();
    double wake_late = std::chrono::duration<double>(Clock::now() - next_t).count() * 1e3;
    double period_ms = last_wake < 0 ? DT * 1e3 : (now - last_wake) * 1e3;
    last_wake = now;
    LocalState s = latest();
    bool cmd_ok = read_block(*g_shm.cmd, c);
    for (int r = 0; r < 10 && !cmd_ok; ++r) { std::this_thread::sleep_for(std::chrono::microseconds(20)); cmd_ok = read_block(*g_shm.cmd, c); }   // a collision with the writer lasts microseconds
    bool fresh = cmd_ok && c.counter != last_counter;
    if (fresh) { last_counter = c.counter; n_cmd_seen++; }
    double fill_ms = 0, crc_ms = 0, write_ms = 0, w2p_ms = -1.0, s2p_ms = -1.0;  // w2p: controller wrote the published command -> end of DDS Write; s2p: newest state arrived -> end of Write
    bool publish_now = false;
    double tgt[N], kp[N], kd[N];

    if (state == ST_IDLE) {
      if (g_signal) go_final(ST_STOPPED, 5, "signal while idle", now);
      else if (fresh && (c.flags & CMD_ENABLE)) {
        std::string why;
        if (now - t_start < 1.5) why = "bridge up less than 1.5 s: the foreign-publisher check is not meaningful yet";
        else if (!s.valid) why = "no valid state";
        else {
          why = check_state(s, now, tb);
          if (why.empty() && publish && g_n_foreign > 0) why = "another publisher is active on rt/lowcmd (" + std::to_string(g_n_foreign.load()) + " msgs): not in debug mode";
          if (why.empty()) { double mx = 0, mt = 0; for (int i = 0; i < N; ++i) if (!absent(i)) { mx = std::max(mx, std::fabs(s.dq[i])); mt = std::max(mt, s.temp[i]); }
            if (mx > 0.5) why = "robot is moving (max |dq| " + std::to_string(mx) + ")"; else if (mt > 60.0) why = "a motor is at " + std::to_string((int)mt) + " C"; }
          if (why.empty()) why = firewall(c, s, prev_target, true, tb);
        }
        if (!why.empty()) go_final(ST_REFUSED, 7, why, now);
        else {
          state = ST_ACTIVE; last_good = c; std::copy(c.target, c.target + N, prev_target); t_last_new = now; rejects = 0;
          set_status(ST_ACTIVE, 0, "active");
          std::printf("ACTIVE\n");
        }
      }
    }
    if (state == ST_ACTIVE) {
      std::string why = check_state(s, now, tb);
      if (g_signal) go_stopping("signal", 5, now);
      else if (cmd_ok && fresh && (c.flags & CMD_END)) go_final(ST_STOPPED, 6, "clean end requested by the controller", now);
      else if (cmd_ok && fresh && (c.flags & CMD_STOP)) go_stopping("stop requested by the controller", 4, now);
      else if (!why.empty()) go_stopping(why, 3, now);
      else {
        if (fresh) {
          t_last_new = now;
          std::string fw = firewall(c, s, prev_target, false, tb);
          if (fw.empty()) { last_good = c; std::copy(c.target, c.target + N, prev_target); rejects = 0; }
          else { rejects++; sp.n_cmd_rejected++; std::printf("firewall rejected command %llu: %s\n", (unsigned long long)c.counter, fw.c_str()); if (rejects >= 3) go_stopping("firewall: " + fw, 2, now); }
        }
        if (state == ST_ACTIVE && (now - t_last_new) * 1e3 > dead_ms) { sp.n_deadman++; go_stopping("dead-man: no new command for " + std::to_string((int)((now - t_last_new) * 1e3)) + " ms", 1, now); }
        if (state == ST_ACTIVE) {
          for (int i = 0; i < N; ++i) { tgt[i] = last_good.target[i]; kp[i] = last_good.kp[i]; kd[i] = last_good.kd[i]; }
          publish_now = true;
        }
      }
    }
    if (state == ST_STOPPING) {
      double u = now - t_stop;
      if (u > fade_s + damp_s) go_final(ST_STOPPED, sp.reason_code, reason + " (stop sequence complete)", now);
      else {
        bool ok = s.valid && finite_all(s.q, N);
        for (int i = 0; i < N; ++i) {
          tgt[i] = (ok && !absent(i)) ? s.q[i] : prev_target[i];
          kp[i] = last_good.kp[i] * std::max(0.0, 1.0 - u / fade_s);
          kd[i] = std::max(last_good.kd[i], tb.kd[i]);
        }
        publish_now = true;
      }
    }
    if (publish_now) {
      for (int i = 0; i < N; ++i) if (absent(i)) { tgt[i] = 0.0; kp[i] = 0.0; kd[i] = 0.0; }
      double a0 = now_s();
      cmd.mode_pr() = 0; cmd.mode_machine() = s.mm;
      for (int i = 0; i < N; ++i) {
        auto& m = cmd.motor_cmd().at(i);
        m.mode() = 1; m.q() = (float)tgt[i]; m.dq() = 0.0f; m.tau() = 0.0f; m.kp() = (float)kp[i]; m.kd() = (float)kd[i];
      }
      double a1 = now_s();
      cmd.crc() = Crc32Core((uint32_t*)&cmd, (sizeof(cmd) >> 2) - 1);
      double a2 = now_s();
      if (publish) pub->Write(cmd);
      double a3 = now_s();
      fill_ms = (a1 - a0) * 1e3; crc_ms = (a2 - a1) * 1e3; write_ms = (a3 - a2) * 1e3;
      if (state == ST_ACTIVE && last_good.t_client > 0.0) w2p_ms = (a3 - last_good.t_client) * 1e3;
      if (s.valid) s2p_ms = (a3 - s.t) * 1e3;
      sp.n_published++; sp.t_last_publish = a3; sp.last_write_ms = write_ms; sp.max_write_ms = std::max(sp.max_write_ms, write_ms);
      v_write.push_back(write_ms);
    }
    sp.n_cmd_seen = n_cmd_seen; sp.t_heartbeat = now; sp.cmd_age_ms = (state == ST_IDLE || t_last_new == 0.0) ? 0.0 : (now - t_last_new) * 1e3;
    write_block(*g_shm.stat, sp);
    v_wake.push_back(wake_late); v_period.push_back(period_ms); v_cmdage.push_back(sp.cmd_age_ms);
    char row[400];
    std::snprintf(row, sizeof row, "%d,%.4f,%u,%llu,%.2f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.3f,%.3f,%llu,%s\n", tick, now - t_start, state, (unsigned long long)c.counter, sp.cmd_age_ms, wake_late, period_ms,
                  s.valid ? (now - s.t) * 1e3 : -1.0, fill_ms, crc_ms, write_ms, w2p_ms, s2p_ms, (unsigned long long)sp.n_cmd_rejected, sp.reason);
    rows.emplace_back(row);
    prev_fresh = fresh; prev_t = now;
    tick++;
    if ((state == ST_STOPPED || state == ST_REFUSED) && now - t_end > linger_s) break;
  }
  if (!outdir.empty()) { std::ofstream f(outdir + "/bridge_log.csv"); for (auto& r : rows) f << r; }
  Stats w = stats(v_write), wk = stats(v_wake), pr = stats(v_period);
  std::printf("END state %u (%s) | ticks %d | published %llu | commands seen %llu rejected %llu | dead-man events %llu | state msgs %llu (CRC mismatches %llu)\n", state, reason.c_str(), tick,
              (unsigned long long)sp.n_published, (unsigned long long)n_cmd_seen, (unsigned long long)sp.n_cmd_rejected, (unsigned long long)sp.n_deadman, (unsigned long long)g_n_msgs.load(), (unsigned long long)g_n_crc_bad.load());
  std::printf("TIMING ms: write mean %.3f p99 %.3f max %.3f | wake lateness mean %.3f p99 %.3f max %.3f | period p99 %.3f max %.3f\n", w.mean, w.p99, w.mx, wk.mean, wk.p99, wk.mx, pr.p99, pr.mx);
  finish(0);
}
