#!/usr/bin/env python3
"""Merges the delay-sweep JSONs of scripts/evaluate_delay.py into tables and finds where each metric leaves its normal band.

    python scripts/summarize_delay.py logs/eval/delay_v11_action.json logs/eval/delay_v11_obs.json logs/eval/delay_v11_stall.json [--md out.md]

Normal band (per metric and scenario): the baseline (extra action delay 0 = the trained model) mean +- max(3 x its between-seed std, 15% of |mean|, a floor):
tilt 0.3 deg, drift 0.5 cm (turn wander 1 cm), torque 1 N*m, chatter 0.01, yaw rate 0.03 rad/s, settle time 0.3 s. The fall rate leaves the band above baseline + 0.02.
"""
import argparse
import json
import math

FLOOR = {"tilt_rms": 0.3, "tilt_max": 0.3, "drift_cm": 0.5, "turn_wander_cm": 1.0, "chatter": 0.01, "wz_reached": 0.03, "settle_s": 0.3}
METRICS = ["fall_rate", "tilt_rms", "tilt_max", "drift_cm", "wz_reached", "turn_wander_cm", "settle_s", "peak_tq_knee", "peak_tq_ankle_pitch",
           "peak_tq_hip_pitch", "peak_tq_waist_yaw", "chatter"]
ORDER = {"action": 0, "obs": 1, "stall": 2}
UNIT = {"action": "ms extra action delay", "obs": "ms observation age", "stall": "ticks stalled (20 ms each), 1% per tick"}


def key_sort(k):
    kind, v = k.split(":")
    return ORDER[kind], int(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    res, meta = {}, {}
    for f in args.files:
        d = json.load(open(f))
        res.update(d["results"])
        meta = {k: d[k] for k in ("checkpoint", "noise", "envs_per_seed_group", "seeds") if k in d}
    keys = sorted(res, key=key_sort)
    scens = list(res[keys[0]].keys())
    base = res["action:0"]
    lines = [f"# Delay sweep ({meta})", ""]
    for scen in scens:
        lines += [f"## Scenario: {scen}", "", "| setting | " + " | ".join(m for m in METRICS if m in base[scen]) + " |", "|---|" + "---|" * len([m for m in METRICS if m in base[scen]])]
        for k in keys:
            r = res[k][scen]
            lines.append(f"| {k} | " + " | ".join(f"{r[m]:.3g}" for m in METRICS if m in base[scen]) + " |")
        lines.append("")
    # band analysis
    lines += ["## First setting that leaves the normal band (per sweep)", ""]
    first = {}
    for kind in ORDER:
        ks = [k for k in keys if k.startswith(kind + ":") and k != "action:0"]
        if not ks:
            continue
        lines.append(f"### {kind}: {UNIT[kind]}")
        lines += ["", "| scenario | metric | baseline | band | first setting outside | value there |", "|---|---|---|---|---|---|"]
        anywhere = None
        for scen in scens:
            for m in METRICS:
                if m not in base[scen]:
                    continue
                b, sd = base[scen][m], base[scen]["_seed_std"].get(m, 0.0)
                if math.isnan(b):
                    continue
                if m == "fall_rate":
                    lo, hi = -1.0, b + 0.02
                else:
                    w = max(3 * (sd if sd == sd else 0.0), 0.15 * abs(b), FLOOR.get(m, 1.0 if m.startswith("peak_tq") else 0.0))
                    lo, hi = b - w, b + w
                out = next((k for k in ks if not (lo <= res[k][scen][m] <= hi) and not math.isnan(res[k][scen][m])), None)
                if out:
                    lines.append(f"| {scen} | {m} | {b:.3g} | {lo:.3g}..{hi:.3g} | {out.split(':')[1]} | {res[out][scen][m]:.3g} |")
                    v = int(out.split(":")[1])
                    anywhere = v if anywhere is None else min(anywhere, v)
        first[kind] = anywhere
        lines += ["", f"**Any metric outside its band from: {anywhere if anywhere is not None else 'none in the swept range'}**", ""]
    # largest setting with every fall rate <= baseline+0.02 and no metric outside the band
    lines += ["## Summary", ""]
    for kind in ORDER:
        ks = [k for k in keys if k.startswith(kind + ":")]
        if not ks:
            continue
        ok = []
        for k in ks:
            good = True
            for scen in scens:
                if res[k][scen]["fall_rate"] > base[scen]["fall_rate"] + 0.02:
                    good = False
            ok.append((k, good))
        last_ok = [k for k, g in ok if g]
        lines.append(f"- {kind}: no extra falls up to **{last_ok[-1].split(':')[1] if last_ok else 'none'}** ({UNIT[kind]}); first band exit: {first.get(kind)}")
    text = "\n".join(lines)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
