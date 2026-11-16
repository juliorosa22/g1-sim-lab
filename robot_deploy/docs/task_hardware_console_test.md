# Task for the hardware session: test the hardware console (camera + mode observer + mode switching)

From the simulation session, 2026-11-16. Read `docs/mode_switching_gui.md` first (what the console is, what the SDK says, what is unknown). The console is code only so far: mock-tested
(`tests/test_modes.py` 22 checks, `tests/test_gui_server.py` 12 checks), **never run against the robot**. Every step below that sends something to the robot needs the user's explicit
yes at that moment (`CLAUDE.md` rules 2, 3 and 7); an earlier yes does not carry over. Read-only steps (A, B) need no yes beyond connecting.

## Before anything
- Robot on the Ethernet cable (`eno2`, 192.168.123.x) or Wi-Fi; `scripts/robot_check.sh` is fine. The laptop needs `unitree_sdk2py` (the checkout at `UnitreeRobot/unitree_sdk2_python`) for `--real`, and for the camera a shell with ROS2 Humble and the CycloneDDS settings of `UnitreeRobot/scripts/init_ros_env.sh` (system Python).
- `unitree-start-joystick.service` stopped on the robot with the user's ok (it returns after every reboot); our own controller NOT running; nobody else publishing `rt/lowcmd`.
- For steps C-E: robot hoisted, feet off the ground, nothing under or beside the limbs, e-stop (or power) in the user's hand, `scripts/robot_estop.sh` ready (`--dry-run` checked first), the robot's own RC in the user's other hand.
- Start the console from the repo root: `python3 robot_deploy/gui/server.py --real --interface eth0` (use the interface that carries 192.168.123.x; `--camera none` if ROS2 is not set up, `--camera ros` is the default with `--real`). Open http://localhost:8770 in the laptop browser. The page starts DISARMED; `--real` only reads until the ARM box is ticked.
- First, with no robot, `python3 robot_deploy/gui/server.py --mock` and click through everything once so the user knows the page.

## Step A (read only): does the console see the robot?
Goal: state, FSM, mode name and camera match reality. Nothing commanded.
1. Robot in Unitree's normal (onboard) mode, on the support. Check on the page: world = ONBOARD, "mode name" non-empty (write it down: it is the name `SelectMode` needs), FSM id and text, tilt about 0-2 deg, `rt/lowcmd` rate (PC1 streams its own commands, so a non-zero rate is expected here), lowstate age a few ms, camera frame age below 1 s.
2. Camera: the RealSense colour image appears and moves with the robot; note the delay and the frame rate; look at the laptop CPU and, with `scripts/robot_ssh.sh 'top -bn1 | head -15'` (read only), the Jetson CPU before and after starting the console (the Jetson was tight at 50 Hz: the console must not add load; it only subscribes).
3. If the page shows "mode unknown" or an error: stop, record the text from the log panel, do not retry blindly.

## Step B (read only, RC in hand): build the FSM table with the mode observer
Goal: learn which FSM id each RC action gives, before any button can set it. The user presses the RC; nothing is sent from the laptop.
Press, one at a time, waiting a few seconds for each, and record the "Mode observer" rows (time, mode name, FSM id, mode_machine) and what the robot did: damp, zero torque, ready / lock stand, balance (F1+Up gave FSM 801 earlier), walk start, return to damp; the L2+R2 debug-mode entry the user used before (the mode name should become empty).
Put the table in `docs/open_issues.md` (date, RC combo, mode name, FSM id, mode_machine, observed behaviour). The id of the "stiff stand" the user wants is in this table; the console's buttons only know 0, 1, 3, 500, 702, 706 (from the SDK), so a new id needs a code change in `gui/modes.py` (`ALLOWED_FSM`) and a test, not a workaround.

## Step C (needs the user's yes, hoisted): soft FSM changes
Goal: `SetFsmId` from software works and is benign.
1. ARM the console (tick ARM). From damp: click FSM 0 (zero torque), then FSM 1 (damp). Watch `tau_est`, `mode_machine`, the observer table, and the robot (limp vs damped: a hand-guided joint resists in damp, not in zero torque).
2. Click DAMP (the red button) from damp and from zero torque: it must not change anything alarming.
3. Do NOT click sit / start / lie-stand / squat-stand in this task (they need "supported" and are a later step with a harness).
Record: how long each change took, any error text, what `rt/lowcmd` rate did.

## Step D (needs the user's yes, hoisted, from damp): release and hand back, no controller running
Goal: answer the open questions 3 and 4 of `mode_switching_gui.md`. This is the risky one: handing back lets Unitree's controller act on the pose we leave.
1. Robot in damp (observer shows FSM 1), hoisted, limbs free. Tick ARM and "robot is hoisted / supported", click "Enter debug mode". Expected: mode name becomes empty within seconds, the remembered name is shown. Record: did the joints go limp (compare `tau_est`), how long, did PC1's `rt/lowcmd` stream stop, any foreign publisher.
2. With `rt/lowcmd` rate 0 (the button is refused otherwise), click "Hand back to Unitree". Expected: mode name returns, FSM 1 (damp), `rt/lowcmd` stream returns. Record the time and any error.
3. If anything is unexpected (stuck in an intermediate mode, the service does not return): do not click again; use the RC (L2+R2 sequence the user knows) to recover, and write what happened in `open_issues.md`.

## Step E (needs the user's yes, hoisted): take-over after our controller
Goal: the user's real aim: if our policy misbehaves, Unitree's controller takes over quickly.
1. Run the v11 hoisted standing as in Stage 2 (`run.py`, existing procedure, its own yes). Then stop it with its own stop sequence (fade + damp), wait until the console shows `rt/lowcmd` rate 0.
2. Click "Hand back to Unitree". Compare with taking over on the RC (the way the user does it today): time, behaviour, what the joints did.
3. Never from a standing pose without a harness or gantry in this task; handing back to a stand/walk FSM is a later task.

## What to report back (in `docs/open_issues.md`, same day)
The mode name; the FSM table from step B; times and behaviour of ReleaseMode / SelectMode; whether the console added Jetson load; camera frame rate and delay; every refusal or error message the console showed (the log panel keeps the last 200 lines); and a recommendation for what the console should offer next. If a button did something different from `mode_switching_gui.md`, fix the doc and the code and add a test in the same change.

## Out of scope for this task
Arm and finger posing (the old `g1_gui/hardware_controller.py` owns `rt/lowcmd` and the Modbus hands in another process; only one publisher may exist at a time), velocity commands, policy start/stop from the page, depth, recording, handing back from a standing pose.
