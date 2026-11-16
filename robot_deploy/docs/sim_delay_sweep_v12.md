# Delay sweep for v12 (walking), simulation, observation noise on

Script: `scripts/evaluate_delay.py --walk` (3 seeds x 64 envs per setting; checkpoint `g1_v12_walk/2026-10-31_22-48-27/model_4998.pt`; raw results `logs/eval/delay_v12_{action,obs,stall}.json` on the laptop).
Trained delay: 0-20 ms action lag (uniform per env, fixed in the episode) plus the 20 ms zero-order hold; no observation delay.
Scenarios: stand 3 s, ramp 1 m/s^2 to vx and hold (15 s); `walk0.8_push` also has 0.3 m/s shoves at 8, 12 and 16 s (20 s).
Entries: fall rate, tilt RMS / max in degrees.

| setting | walk 0.5 | walk 0.8 | walk 1.0 | walk 0.8 + shoves |
|---|---|---|---|---|
| extra action delay 0 ms | 0.0% 2.1/4.5 | 0.0% 2.4/4.8 | 0.0% 2.5/5.0 | 0.0% 2.4/5.1 |
| +10 ms | 0.0% 2.3/4.8 | 0.0% 2.5/5.1 | 0.0% 2.6/5.3 | 0.0% 2.6/5.5 |
| +20 ms | 0.0% 2.5/5.3 | 0.0% 2.7/5.6 | 0.0% 2.8/5.8 | 0.0% 2.8/5.9 |
| +30 ms | 0.5% 2.9/6.0 | 0.0% 3.0/6.3 | 0.0% 3.1/6.6 | 0.5% 3.1/6.7 |
| +40 ms | 1.6% 3.3/7.4 | 3.6% 3.4/7.7 | 8.3% 3.6/8.6 | 7.8% 3.5/8.4 |
| +60 ms | 47.9% 4.3/10.9 | 50.5% 4.6/12.4 | 58.9% 4.6/12.5 | 63.0% 4.5/12.3 |
| observation age 10 ms | 0.0% | 0.0% | 0.0% | 0.0% |
| 20 ms | 0.0% | 0.0% | 0.0% | 0.5% |
| 30 ms | 0.5% | 1.0% | 0.0% | 0.5% |
| 40 ms | 2.6% | 3.1% | 4.7% | 5.2% |
| 60 ms | 46.4% | 44.8% | 56.8% | 58.3% |
| late ticks (1% per tick), 1 tick (20 ms) | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 ticks | 0.0% | 0.0% | 0.5% | 0.0% |
| 3 ticks | 0.0% | 0.0% | 0.5% | 0.0% |
| 5 ticks (100 ms) | 0.0% | 0.5% | 1.6% | 1.0% |

## Budget for v12
- Up to +20 ms extra action delay or observation age: no extra falls, tilt up by about 0.3 deg. Up to +30 ms: at most 0.5-1% falls.
- +40 ms: 2-8% falls (worse at higher speed and with shoves); +60 ms: about half of the runs fall. Do not exceed +30 ms in total extra.
- Occasional late ticks (a stalled controller re-sending the last target) up to 5 ticks (100 ms) at 1% per tick: at most 1.6% falls at 1.0 m/s.
- Compare v11: baseline-like to +10 ms, no extra falls to +20 ms, 17-21% falls under pushes at +40 ms, 27-32% (turns) and 60% (pushes) at +60 ms. v12 has a similar edge (+30 to +40 ms).
- The measured pipeline on the robot (hoisted): state age at the controller ~0.6 ms, policy ~1.2 ms (Jetson, numpy), controller-to-publish 0.34 ms with the on-arrival bridge (5.3 ms with the fixed tick), joint response 3-8 ms after the command: about 10-15 ms in total, inside the trained 0-20 ms plus margin.
