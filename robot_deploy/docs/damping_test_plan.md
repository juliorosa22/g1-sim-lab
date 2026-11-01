# Damping test plan (hoisted, debug mode): does the stop sequence really damp?

Status: **Test A PASSED (right elbow, 2026-11-01, see open_issues.md); Test B not run.** Every step that publishes `rt/lowcmd` needs the user's explicit yes for that exact run (`CLAUDE.md` rule 2).

## Why
The controller's stop sequence (after the end of a run, any safety trip, a STOP file or Ctrl-C) is: kp fades to 0 over 0.5 s while the target follows the measured pose, then kd-only damping for 3 s, then publishing ends.
Everything in the later stages assumes the damping step actually resists motion. Measured so far: the Jetson's commands act as a PD in debug mode (hold tests, steps, ramps), but a **damping-only command (kp = 0, kd > 0, tau = 0) was only tried in the default zero-torque state, where PC1's own stream overrode it and it had no visible effect.** It has never been tried in debug mode.
Also measured: when our publishing ends, all motors fall back to mode 0 (limp). So damping is the only soft landing there is, and a joint that is NOT damped just falls.

## Preconditions (every run)
Robot hoisted, feet off the ground, nothing under or beside the limbs; e-stop or power in your hand; `unitree-start-joystick.service` stopped (it is); the robot in Unitree's **debug mode, entered by you on the RC** (L2+R2 sequence from the `deploy_real` README); `jointtest.sh check` / `run.py --plan` report **no other publisher** on `rt/lowcmd` (today's dry run correctly refuses: PC1 is still publishing). The read-only full-rate recorder (`record_walk.sh`) runs alongside.
Before Test B the robot copy of the controller must be updated (it lacks the stop-test mode and the start-pose guard): `robot_sync.sh --apply` (dry run first, your yes).

## Test A: one joint, hand-guided (small, first)
**What:** the right elbow (joint 25) gets a damping-only command: kp 0, kd 3.0 (the controller's arm value), tau 0, for 22 s at 50 Hz. All other joints are held in place at half their nominal gains (as in the earlier hold tests). You move the right forearm by hand.
**Command:** `jointtest.sh run 25 hold --kp 0 --kd 3.0 --hold-s 20 --rate 50 --max-exc 1.0 --tau-cap 8 --vel-cap 3` (after the dry run shown to you and your yes). The excursion limit is 1.0 rad (abort at 1.5), because you are meant to move it; speed cap 3 rad/s; torque abort at 12 N*m; a held neighbour drifting more than 0.15 rad also aborts, so touch only the forearm.
**What you do:** once the limbs stiffen (the held joints), grip the right forearm and flex and extend the elbow slowly for ~8 s, then faster for ~8 s, stopping now and then. Tell me what it felt like: like moving through honey, springy, free, or locked.
**Expected if damping works:** `tau_est` follows -kd x speed (slope about -3 N*m per rad/s, 30% tolerance, correlation below -0.8); the arm resists fast motion more than slow; it stays where you leave it (no spring back).
**Fails if:** slope near 0 and the arm feels free (damping does not act from the Jetson), or it feels springy (kp is not 0 in effect).
**Risk:** low. Commanded torque is only viscous; the arm is yours to hold. Worst case the elbow acts as designed or moves freely.

## Test B: the controller's real stop sequence, whole robot, hoisted
**What:** `run.py --publish --stage stoptest --stop-test --stop-test-hold-s 4` (with `CONTROLLER_CONFIRM=yes-stoptest`, after the plan is shown and you say yes). Timeline: 0-2 s all joints enabled, holding the pose they are in, gains rising 30% to 100%; 2-4 s full gains (legs 100/100/100/150/60/60, waist yaw 200, arms 60) at zero error; 4.0-4.5 s kp fades to 0; 4.5-7.5 s kd-only damping (legs 4/4/4/5/3/3, waist 6, arms 3); then publishing ends and the motors go limp. No ramp, no policy, no default pose: the target IS the measured pose.
**Safety in the run:** the controller's own trips (tilt 25 deg, joint speed 5 rad/s, torque caps, tracking error 0.30 rad, stale state, STOP file ...) all lead to the same stop sequence; the start-pose guard does not apply (nothing is ramped).
**Expect and look for:** (1) no motion above ~0.01 rad during the hold; (2) no jerk at the kp fade (speeds below ~1 rad/s); (3) during damping, speeds stay low (below ~1.5 rad/s) and `tau_est` follows -kd x speed on the joints that move; (4) when publishing ends, the limbs fall freely: speeds rise sharply, which is the contrast that proves the damping was acting; (5) the time from the last command to mode 0.
**Fails if:** the limbs accelerate during the "damping" window as they do after it ends, or the fade causes a jump.
**Risk:** moderate (the first time every joint is enabled at full gains, and the end of the run releases all limbs). Zero error at start so no motion is commanded; stand clear when it ends because arms and legs swing down.

## After the tests
I pull the logs and recording, and report slope, correlation, peak speeds and how it felt in `measurements/<date>/damping/REPORT.md` and `docs/open_issues.md`.
- **Both pass:** the stop sequence is real; next are the pose reset by hand, the torque-cap review and the Stage 2 plan.
- **Either fails:** do not go to Stage 2 as designed; options are a longer fade that keeps holding position, a hoist-only stop with the controller never relying on damping, or asking Unitree how damping from the Jetson is meant to work.

## Order of approvals (each separate)
1. `robot_sync.sh --apply` (new `run.py`). 2. You enter debug mode. 3. Test A dry run shown, then the run. 4. Test B plan (`--plan`) shown, then the run.
