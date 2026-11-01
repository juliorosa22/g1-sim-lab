# controller_cpp/: C++ pieces (built ON the robot against Unitree's installed SDK)

Why: the Python publish path (fill 29 motors, checksum, DDS write) takes 2.6 ms median, 7.9 ms p99 and stalled up to 32 ms once in a while (`docs/open_issues.md`), and Python's DDS receive uses ~85% of a core.
Question being answered first: does the C++ SDK avoid those stalls, or does the DDS write itself block (in which case C++ would not help)?

## probe/stoptest_probe.cpp
C++ counterpart of `controller/run.py --stop-test`: hold the measured pose, kp fade, kd-only damping, publishing ends. Same gains, safety checks and sequence; no loop-period trip (it is measured).
Per tick it logs wake-up lateness, period, and the separate cost of the message fill, the checksum and the DDS `Write`, plus the state age; it also verifies the checksum of every incoming state.
```
cd controller_cpp && mkdir -p build && cd build && cmake .. && make -j4     # on the robot (g++ 9.4, cmake 3.16, find_package(unitree_sdk2))
./stoptest_probe --plan                                                     # read-only: preflight, checksum check, prints the plan
./stoptest_probe --out /home/unitree/g1_recordings/cpp/dry1                 # DRY RUN: builds commands incl. fill + checksum, sends nothing
CONTROLLER_CONFIRM=yes-<stage> ./stoptest_probe --publish --stage <stage> --out DIR    # *** publishes rt/lowcmd ***, only after the user's yes for THIS run
```
Publishing needs `--publish`, `--stage` and the matching `CONTROLLER_CONFIRM`; robot hoisted, debug mode, e-stop in hand, `unitree-start-joystick.service` stopped (CLAUDE.md).

## Laptop test (no robot, no SDK)
`tests/run_stub_test.sh` compiles the probe against `stubs/` (a stand-in for the SDK headers that feeds a quiet robot at 1 kHz) and checks: plan, dry run sends nothing, the three publish refusals, a simulated publish, SIGTERM and a STOP file going through the stop sequence.
The stubs mirror the accessors I read in the real headers; the real compile happens on the robot, so small API differences may still show up there.

## bridge/g1_bridge.cpp: DDS owner for the Python controller (laptop-tested only)
Design and protocol: `bridge/DESIGN.md`, `bridge/shm_protocol.h`; Python side `controller/bridge_client.py`, used by `controller/run.py --source bridge`.
```
./g1_bridge                                                      # shadow mode: reads state, publishes nothing, shows every firewall decision in STATUS
CONTROLLER_CONFIRM=yes-<stage> ./g1_bridge --publish --stage <stage>    # *** publishes rt/lowcmd ***, only after the user's yes for THIS run
```
One run per process; a dead controller triggers the fade and damping within ~80 ms. Laptop tests: `python3 tests/test_bridge.py` (run alone, not in parallel; `G1_BRIDGE_SRC=file.cpp` tests a mutated copy). Not yet synced, built or run on the robot.
