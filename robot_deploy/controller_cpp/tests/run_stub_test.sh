#!/usr/bin/env bash
# Laptop test of probe/stoptest_probe.cpp against the stand-in SDK (stubs/). Nothing touches a robot or a network.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"; BIN=/tmp/probe_stub_$$; OUT=$(mktemp -d); FAIL=0
ok() { echo "  PASS $1"; }; bad() { echo "  FAIL $1"; FAIL=1; }
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"$HERE/stubs" "$HERE/probe/stoptest_probe.cpp" -o $BIN -pthread || { echo "COMPILE FAILED"; exit 1; }
echo "== plan (preflight only)";    P=$($BIN --plan); echo "$P" | grep -q "PLAN ONLY" && ok "plan prints and exits" || bad "plan"; echo "$P" | grep -q "CRC mismatches 0" && ok "state checksum verified against the stand-in" || bad "state crc"
echo "== dry run (sends nothing)";   D=$($BIN --out $OUT/dry); echo "$D" | grep -q "reason: finished (stop test)" && ok "dry run finishes" || bad "dry run end"; echo "$D" | grep -q "writes ok 0 failed 0" && ok "dry run wrote nothing" || bad "dry run wrote"
TICKS=$(echo "$D" | sed -n 's/.*ticks \([0-9]*\).*/\1/p'); [ "$TICKS" -ge 370 ] && [ "$TICKS" -le 385 ] && ok "tick count $TICKS (~375 for 4 s hold + 3.5 s stop)" || bad "ticks $TICKS"
python3 - "$OUT/dry/log.csv" <<'PY' && ok "log phases and gains as designed" || bad "log content"
import sys, csv
rows=list(csv.DictReader(open(sys.argv[1]))); kp=[float(r["kp_knee"]) for r in rows]; ph=[int(r["phase"]) for r in rows]
assert abs(kp[0]-45*(0.3 if False else 1))<20, kp[0]            # starts near 30% of 150 = 45
assert max(kp)<=150.001 and kp[-1]==0.0, (max(kp),kp[-1])      # never above full, ends at 0
i=ph.index(4); assert all(kp[j]<=kp[j-1]+1e-9 for j in range(i+1,len(kp))), "kp rises after the stop"
assert abs(kp[i]-150)<1, kp[i]                                 # fades from full (the hold reached full gains)
assert ph.count(1)>=195 and ph.count(4)>=170
PY
echo "== publishing is refused without the confirmation"
$BIN --publish >/dev/null 2>&1; [ $? -eq 2 ] && ok "refused: no --stage" || bad "no stage"
$BIN --publish --stage x >/dev/null 2>&1; [ $? -eq 2 ] && ok "refused: no CONTROLLER_CONFIRM" || bad "no confirm"
CONTROLLER_CONFIRM=yes-y $BIN --publish --stage x >/dev/null 2>&1; [ $? -eq 2 ] && ok "refused: confirm names another stage" || bad "wrong stage"
echo "== simulated publish (stand-in publisher counts writes)"
R=$(CONTROLLER_CONFIRM=yes-t $BIN --publish --stage t --out $OUT/pub); echo "$R" | grep -q "reason: finished (stop test)" && ok "publish run finishes" || bad "publish end"
W=$(echo "$R" | sed -n 's/.*writes ok \([0-9]*\) failed.*/\1/p'); [ "$W" -ge 370 ] && [ "$W" -le 385 ] && ok "writes $W = hold + stop only" || bad "writes $W"
echo "== SIGTERM mid-run goes through the stop sequence"
( $BIN --out $OUT/sig > $OUT/sig.txt & PID=$!; sleep 2.5; kill -TERM $PID; wait $PID ); grep -q "reason: signal" $OUT/sig.txt && ok "signal -> stop sequence" || bad "signal"
python3 - "$OUT/sig/log.csv" <<'PY' && ok "after the signal kp fades to 0 and the run ends within ~3.5 s" || bad "signal log"
import sys, csv
rows=list(csv.DictReader(open(sys.argv[1]))); ph=[int(r["phase"]) for r in rows]; kp=[float(r["kp_knee"]) for r in rows]
i=ph.index(4); n=len(rows)-i
assert 160<=n<=185, n
assert all(kp[j]<=kp[j-1]+1e-9 for j in range(i+1,len(kp))) and kp[-1]==0.0
PY
echo "== STOP file"
mkdir -p $OUT/stopf; ( $BIN --out $OUT/stopf > $OUT/stopf.txt & PID=$!; sleep 2; touch $OUT/stopf/STOP; wait $PID ); grep -q "reason: STOP file" $OUT/stopf.txt && ok "STOP file -> stop sequence" || bad "stop file"
rm -rf $OUT $BIN; echo; [ $FAIL -eq 0 ] && echo "RESULT: ALL PASS" || echo "RESULT: FAILURES"; exit $FAIL
