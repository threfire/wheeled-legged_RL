#!/usr/bin/env python3
"""Compare CSV output from compare_dogleg_collision_motion.py runs."""

import argparse
import csv
import statistics


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--legacy", required=True, help="CSV from --asset legacy")
parser.add_argument("--current", required=True, help="CSV from --asset current")
args = parser.parse_args()


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows, phase):
    subset = [r for r in rows if r["phase"] == phase]
    if not subset:
        return None
    cols = (
        "left_wheel_contact_peak_N", "right_wheel_contact_peak_N",
        "max_undesired_contact_N",
        "left_wheel_slip_speed_m_s", "right_wheel_slip_speed_m_s",
        "base_vx_m_s", "base_yaw_rate_rad_s", "base_z_m",
        "closure_residual_max_mm", "reward_track_height_tight",
        "reward_wheel_air_spin", "reward_undesired_contact",
    )
    result = {k: statistics.mean(float(r[k]) for r in subset) for k in cols}
    result["closure_residual_max_mm"] = max(float(r["closure_residual_max_mm"]) for r in subset)
    result["max_undesired_contact_N"] = max(float(r["max_undesired_contact_N"]) for r in subset)
    result["samples"] = len(subset)
    return result


old, new = load(args.legacy), load(args.current)
print("Comparison: CURRENT minus LEGACY; same commanded motion, fixed reset, randomization disabled.")
for phase in ("settle", "ramp", "drive", "coast"):
    a, b = summarize(old, phase), summarize(new, phase)
    if a is None or b is None:
        print(f"{phase}: missing phase data (legacy={None if a is None else a['samples']}, current={None if b is None else b['samples']})")
        continue
    print(f"\n[{phase}] samples legacy/current={a['samples']}/{b['samples']}")
    for key in a:
        if key == "samples":
            continue
        print(f"{key:34s} legacy={a[key]: .5f} current={b[key]: .5f} delta={b[key]-a[key]:+.5f}")
