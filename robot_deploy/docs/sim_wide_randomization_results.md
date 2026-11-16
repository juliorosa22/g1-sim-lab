# Wider plant randomisation: v11b and v12b against v11 and v12 (simulation, 2026-11-16)

What changed in training (`g1_rl/wide_randomization.py`, tasks `G1-Walk-Flat-Inspire-V11b-v0` / `V12b`): floor friction down to 0.15 (was 0.3), leg + waist-yaw stiffness and damping x0.6-1.4 drawn independently (was x0.8-1.2),
all link masses x0.8-1.3 on top of the torso +-5 kg, torso COM +-5 cm (was +-3), torque limit x0.7-1.0 (was 1.0). Warm start from v11 (model_5997, 1000 iterations -> model_6996) and v12 (model_4998, 1500 iterations -> model_6497),
final reward weights from the first iteration. Checkpoints on the lab: `logs/rsl_rl/g1_v11b_wide/2026-11-16_06-58-08/model_6996.pt`, `logs/rsl_rl/g1_v12b_wide/2026-11-16_07-43-33/model_6497.pt` (not exported).
All numbers below: the new checkpoints evaluated on the ORIGINAL Play tasks (the old randomisation), so every row compares the same plant. Observation noise on, 8 envs per cell: differences below about 5 points are noise.

## Nominal behaviour
- **v11b vs v11**: 100% survival in both. Standing and turning sway a little more (turn-settle with noise: 39.8 vs 33.2 mm/s, drift 1.4 vs 1.0 cm); yaw -0.5 rad/s is tracked worse (-0.35 achieved vs -0.51);
  command grid in the recommended box: 0.8% falls with reversal or the stressed run (v11 0%); tracking error in the box 0.095 vs 0.054 m/s; a reversal from vx 0.8 falls 62% vs 27%. Slightly worse.
- **v12b vs v12**: the same: in-box falls 0% in both, reversal falls equal (29% vs 27% from 1.0 m/s), stop test equal (speed 0.4625 vs 0.464 m/s at 0.5, settled 49% vs 54%).

## Plant sweep: fall % over the in-range command box
| setting | v11 | v11b | v12 | v12b |
|---|---|---|---|---|
| baseline | 0.6 | 1.3 | 0.3 | 0.0 |
| mass x0.85 | 3.1 | 7.5 | 0.3 | 2.2 |
| mass x1.15 | 0.6 | 0.6 | 1.1 | 0.6 |
| mass x1.3 | 1.3 | 1.0 | 4.2 | 0.0 |
| torso +8 kg | 1.0 | 0.0 | 0.8 | 0.0 |
| friction x0.4 | 9.0 | 9.8 | 1.9 | 5.0 |
| friction x0.2 | 54.8 | 55.4 | 32.5 | 41.9 |
| friction x1.5 | 2.7 | 4.2 | 1.9 | 2.8 |
| kp x0.6 | 18.5 | 0.6 | 25.8 | 0.6 |
| kp x1.4 | 17.5 | 22.9 | 9.7 | 17.2 |
| kd x0.5 | 8.5 | 15.2 | 3.1 | 10.8 |
| kd x1.6 | 1.3 | 1.9 | 15.0 | 8.1 |
| kp,kd x0.7 | 0.8 | 0.6 | 3.3 | 0.3 |
| kp,kd x1.3 | 6.0 | 12.3 | 1.4 | 6.7 |
| torque x0.9 | 1.3 | 2.5 | 0.3 | 0.8 |
| torque x0.75 | 1.0 | 2.9 | 0.3 | 0.3 |
| torque x0.6 | 3.8 | 0.4 | 5.6 | 0.0 |
| jfric 1 Nm | 1.0 | 2.3 | 0.0 | 0.3 |
| jfric 2 Nm | 0.6 | 1.5 | 0.3 | 0.3 |
| jfric 4 Nm | 0.8 | 1.0 | 0.6 | 0.3 |
| delay +10 ms | 1.3 | 3.8 | 0.0 | 0.8 |
| delay +20 ms | 2.3 | 2.3 | 0.0 | 1.4 |
| delay +30 ms | 0.4 | 3.1 | 0.0 | 0.3 |
| com +4 cm | 0.8 | 3.5 | 0.0 | 1.4 |
| com -4 cm | 1.7 | 1.9 | 0.3 | 0.8 |
| combined mild | 1.7 | 1.5 | 0.8 | 0.3 |
| combined hard | 54.0 | 9.4 | 46.7 | 8.9 |

Mean over the 26 perturbed settings: v11 7.5% -> v11b 6.5%; v12 6.0% -> v12b 4.3%.

## What it says
- **Much better with a soft or weak plant**: kp x0.6 (v11 18.5 -> 0.6, v12 25.8 -> 0.6), torque x0.6 (3.8 -> 0.4, 5.6 -> 0.0), mass x1.3 on v12 (4.2 -> 0.0), and the combined hard case (v11 54 -> 9.4, v12 47 -> 8.9).
- **Worse with a stiff or lightly damped plant**: kp x1.4 (v11 17.5 -> 22.9, v12 9.7 -> 17.2), kd x0.5 (8.5 -> 15.2, 3.1 -> 10.8), kp and kd together x1.3 (6.0 -> 12.3, 1.4 -> 6.7), v11 mass x0.85 (3.1 -> 7.5). The wide training moved the policies toward assuming softer joints.
- **Floor friction did not improve** (x0.2: 55 -> 55 and 32 -> 42; x0.4: 9 -> 10 and 2 -> 5), although the training range went down to 0.15: x0.4 and x0.2 of the old range (0.12-0.48 and 0.06-0.24) are mostly below it. A policy that has to walk on near-ice is a different problem
  (it would need slower gaits or slip-aware rewards), not more of the same randomisation. The floor friction on the robot's floor has to be measured.
- The measured actuator on the robot (hoisted tests, onboard-controller recordings) follows the PD torque with slope 0.92-0.99 and a 3-7% torque shortfall: the real plant leans to the SOFT side (kp x0.93-0.97), which is the side v11b/v12b handle better, not the stiff side where they got worse.

## Recommendation
- **Keep v11 and v12 as the hardware candidates.** v11b is slightly worse in nominal quality (more sway, worse tracking, worse at command reversal) and v12b is the same nominally; neither fixes the friction weakness.
- **Keep v11b / v12b as ready alternatives**, to be exported and tried if the hardware shows a soft or weak plant (low effective stiffness, torque shortfall beyond 10%, heavy payloads with the arms).
- The next sim step that matters more is a policy or a speed limit for LOW FRICTION, after the hardware session reports the floor's friction. A middle option, if wanted: retrain with the soft side only (stiffness x0.6-1.0, torque x0.7-1.0, mass x0.9-1.3) and keep the old friction range.
