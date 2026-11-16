# Hardware console (camera + mode switching): what it is, what is known, what must be discovered on the robot

Written 2026-11-16 by the simulation session. Code: `gui/modes.py` (logic and SDK adapter), `gui/camera.py`, `gui/server.py`, `gui/static/index.html`,
tests `tests/test_modes.py` (22 checks, mock robot, all pass). **Nothing here has been run against the robot.** Every sequence below that touches modes is an
UNVERIFIED hypothesis until a hoisted test with the user's explicit yes (`CLAUDE.md` rules 2 and 3 apply to each run).

## Run
    python3 robot_deploy/gui/server.py --mock                      # no robot: mock backend + test-pattern camera, http://localhost:8770
    python3 robot_deploy/gui/server.py --real --interface eth0     # the robot (user's yes for that run); camera: --camera ros (default with --real) | none | URL
`--real` only READS (CheckMode, GetFsmId, rt/lowstate, a read-only subscription on rt/lowcmd) until the page is ARMED. The server never publishes motor commands.
Camera: `ros` subscribes to `/camera/camera/color/image_raw` (the topic of `UnitreeRobot/scripts/g1_camera_viewer.py`); start the server from a shell with ROS2 Humble and the
CycloneDDS settings of `UnitreeRobot/scripts/init_ros_env.sh` (system Python). It only subscribes, so the robot's CPU load is the camera node's own (earlier measurements:
the Jetson is tight at 50 Hz, so do not start extra encoders on it). Frames are downscaled to 640 px and re-encoded on the laptop (10 fps).

## What the SDK says (unitree_sdk2py in this checkout)
- `MotionSwitcherClient` (service "motion_switcher"): `CheckMode()` -> `{"name": ...}`, `SelectMode(name_or_alias)`, `ReleaseMode()`. `hardware_controller.py` of the old GUI releases the
  onboard service with `while CheckMode()["name"]: ReleaseMode()` and then publishes `rt/lowcmd` itself: that is "debug mode" from software.
- `LocoClient` (service "sport", works only while the onboard service runs): `SetFsmId(id)`; ids in the SDK: 0 zero torque, 1 damp, 3 sit, 500 start, 702 lie->stand, 706 squat<->stand;
  `GetFsmId()`. The robot was seen in FSM 801 (balance) after the RC's F1+Up (`open_issues.md`, 2026-10-29).
- Hardware facts to respect: when our publishing ends all motors fall to mode 0 (limp); a second publisher on `rt/lowcmd` competes with the onboard stream; `unitree-start-joystick.service`
  must be stopped for policy tests.

## What the console does and refuses (`modes.py`, covered by the tests)
| action | allowed when | what it does |
|---|---|---|
| DAMP | always (no arming), onboard service running | `SetFsmId(1)`. In debug mode it refuses and points to the controller's stop sequence / `scripts/robot_estop.sh` |
| Enter debug mode | armed, "supported" declared, onboard FSM is 0 or 1 | remembers the onboard mode name, `ReleaseMode()` until `CheckMode()` is empty |
| Hand back to Unitree | armed, `rt/lowcmd` rate known and 0 (our controller stopped) | `SelectMode(remembered name)`, waits for it, then `SetFsmId(1)` (damp). Never to a stand or walk |
| Select mode by name | same | the same, for a name typed by the user (when the GUI started in debug mode) |
| Onboard FSM buttons | armed; 0 and 1 freely, 3/500/702/706 only with "supported" | `SetFsmId`; 801 and any other id are refused (observed only) |
Everything is logged on the page. The page starts disarmed.

## What has to be discovered (read-only first, nothing commanded)
1. **The mode name.** `CheckMode()` on this robot while Unitree's controller runs (RC, normal mode): the console shows it under "mode name". The hand-back uses the name remembered at `enter_debug`; if the
   console starts already in debug mode (as in the hardware session, entered with L2+R2), the name must be typed once. Get it by watching the observer after putting the robot back with the RC.
2. **The RC sequences as FSM ids.** The "Mode observer" table logs every change of (mode name, FSM id, mode_machine) while someone presses the RC (damp, ready/lock stand, balance, walk). This builds the
   id table without sending anything, and tells which FSM id is the "stiff stand" (the user's goal) before any button is allowed to set it.
3. **Does `ReleaseMode()` from software behave like L2+R2?** Does the robot go limp, what does `tau_est` do, how long does the service take, does `rt/lowcmd` from PC1 stop?
4. **Does `SelectMode(name)` bring the onboard controller back, in which FSM, and how hard does it act on a pose we left it in?** This is the dangerous one: the onboard stand can drive the joints
   from the pose we left. First tests: hoisted, in damp, with the e-stop in hand, and our controller's pose near the default.

## Suggested staged test (each step needs the user's explicit yes at the time; all hoisted, robot supported)
0. `--real`, disarmed: confirm state, FSM, mode name, camera, `rt/lowcmd` rate against the RC's own readings; press RC combos and fill the observer table. Nothing commanded.
1. Armed, onboard service running, FSM damp: `set_fsm(0)` and `set_fsm(1)` (soft states) and watch `tau_est`/`mode_machine`.
2. From damp: Enter debug mode; check the limpness and that CheckMode is empty; then Hand back (no controller running) and check the mode name and FSM 1.
3. With our controller in the loop (run.py, the stage 2 hoisted standing): stop it with its own stop sequence, then Hand back; compare with the RC takeover (L1/L2+B) the user knows.
4. Only then think about handing back from a standing pose; never without a harness or gantry.

## Not in this console yet
Arm and finger posing (`UnitreeRobot/g1_gui/hardware_controller.py` owns `rt/lowcmd` and Modbus hands in its own process: only one publisher may exist at a time, so merging needs its own design),
velocity commands, policy start/stop, depth view, recording. The camera feed is also the natural input for the arm-imitation work (a pose model on the laptop GPU); not started.
