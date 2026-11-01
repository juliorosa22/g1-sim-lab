// Shared-memory protocol between g1_bridge (C++) and the controller. See DESIGN.md. Fixed layout, 8-byte aligned, no padding between members (static_asserts below).
// The Python mirror is controller/bridge_client.py; tests/test_bridge.py checks both against `g1_bridge --print-layout`.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace g1bridge {
constexpr uint32_t MAGIC = 0x52423147;  // "G1BR" little-endian
constexpr uint32_t VERSION = 1;
constexpr int N = 29;

enum BridgeState : uint32_t { ST_IDLE = 0, ST_ACTIVE = 1, ST_STOPPING = 2, ST_STOPPED = 3, ST_REFUSED = 4 };
enum CmdFlags : uint32_t { CMD_ENABLE = 1, CMD_STOP = 2, CMD_END = 4 };

struct Header {
  uint32_t magic, version, n_motor, header_size;
  uint32_t off_state, off_cmd, off_status, total_size;
};

struct StatePayload {
  double t_arrival;  // CLOCK_MONOTONIC seconds when the newest rt/lowstate arrived at the bridge
  uint64_t n_msgs, n_crc_bad, n_foreign_cmd;
  uint32_t tick, mode_machine;
  double q[N], dq[N], tau[N], temp[N];
  int32_t mode[N + 1];  // 30 entries: keeps the payload a multiple of 8 bytes; the last one is unused
  double gyro[3], quat[4];
};
struct CmdPayload {
  uint64_t counter;  // must advance on every new command; the dead-man watches it
  double t_client;   // CLOCK_MONOTONIC seconds at the controller
  uint32_t flags, pad;
  double target[N], kp[N], kd[N];
};
struct StatusPayload {
  uint32_t state, reason_code;
  char reason[96];
  uint64_t n_published, n_cmd_seen, n_cmd_rejected, n_deadman;
  double t_started, t_heartbeat, t_last_publish, last_write_ms, max_write_ms, cmd_age_ms;
  uint32_t publishing, pad;
};

template <class P> struct Block {
  std::atomic<uint64_t> seq;  // odd while the writer is inside
  uint64_t chk_xor, chk_sum;  // XOR and wrapping sum of the payload's 64-bit words
  P p;
};
using StateBlock = Block<StatePayload>;
using CmdBlock = Block<CmdPayload>;
using StatusBlock = Block<StatusPayload>;

constexpr size_t align64(size_t x) { return (x + 63) & ~size_t(63); }
constexpr size_t OFF_STATE = align64(sizeof(Header));
constexpr size_t OFF_CMD = OFF_STATE + align64(sizeof(StateBlock));
constexpr size_t OFF_STATUS = OFF_CMD + align64(sizeof(CmdBlock));
constexpr size_t TOTAL = OFF_STATUS + align64(sizeof(StatusBlock));

static_assert(sizeof(Header) == 32, "header layout");
static_assert(sizeof(StatePayload) % 8 == 0 && sizeof(CmdPayload) % 8 == 0 && sizeof(StatusPayload) % 8 == 0, "payloads are whole 64-bit words");
static_assert(sizeof(StatePayload) == 8 + 24 + 8 + 4 * N * 8 + (N + 1) * 4 + 7 * 8, "state payload has no padding");
static_assert(sizeof(CmdPayload) == 8 + 8 + 8 + 3 * N * 8, "command payload has no padding");
static_assert(sizeof(StatusPayload) == 8 + 96 + 32 + 48 + 8, "status payload has no padding");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "seq must be lock-free to live in shared memory");

inline void checksum(const void* p, size_t bytes, uint64_t& x, uint64_t& s) {
  const uint64_t* w = static_cast<const uint64_t*>(p);
  x = 0; s = 0;
  for (size_t i = 0; i < bytes / 8; ++i) { x ^= w[i]; s += w[i]; }
}

// single writer per block
template <class P> void write_block(Block<P>& b, const P& payload) {
  uint64_t s = b.seq.load(std::memory_order_relaxed);
  b.seq.store(s + 1, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  std::memcpy(&b.p, &payload, sizeof(P));  // readers may overlap this copy by design: they check seq and the checksums
  uint64_t x, sm;
  checksum(&payload, sizeof(P), x, sm);
  b.chk_xor = x; b.chk_sum = sm;
  std::atomic_thread_fence(std::memory_order_release);
  b.seq.store(s + 2, std::memory_order_release);
}
// returns true only for a consistent copy
template <class P> bool read_block(const Block<P>& b, P& out) {
  for (int tries = 0; tries < 8; ++tries) {
    uint64_t s1 = b.seq.load(std::memory_order_acquire);
    if (s1 & 1) continue;
    std::memcpy(&out, &b.p, sizeof(P));
    uint64_t x = b.chk_xor, sm = b.chk_sum;
    std::atomic_thread_fence(std::memory_order_acquire);
    uint64_t s2 = b.seq.load(std::memory_order_relaxed);
    if (s1 != s2) continue;
    uint64_t cx, cs;
    checksum(&out, sizeof(P), cx, cs);
    if (cx == x && cs == sm) return true;
  }
  return false;
}
}  // namespace g1bridge
