#!/usr/bin/env python3
"""Live playground: pick a policy, send velocity commands, move the arms, shove it, all from a web page.

    python scripts/play_policy.py [--port 8765] [--policy v10]       (laptop: scripts/play_policy_local.sh)

then open http://localhost:<port>. This process is only the web server and process manager (no Isaac import). The simulation runs in a
worker process (scripts/play_policy_worker.py), one per policy: choosing another policy in the page stops the worker and starts a new one
(about a minute, the page shows "loading"); the command, arm targets and safe-range lock are carried over.

Page: policy and plant-gain selectors, live camera, telemetry (base height, tilt, commanded vs achieved velocity, foot lifts, falls),
velocity sliders with presets and keyboard drive (W/S forward-back, A/D left-right, Q/E turn, space stops), optional ramp at the
controller's acceleration limits and the 0.05 deadband, shoves, reset, camera views, and the 14 arm-joint sliders with the safe-range lock
(see scripts/play_arms.py, which stays as the arms-only version for v10).

Policies are listed when their checkpoint exists on this machine (logs/lab_checkpoints/<name>/ or the usual logs/rsl_rl runs); either rsl-rl
checkpoint format works. The lab machine has them all; copy the ones you want to the laptop with scp.
"""
import argparse
import atexit
import glob
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO_ROOT, "scripts", "play_policy_worker.py")

# id, label, Play task, kind (what the policy is for), checkpoint candidates (first that exists), does the plant-gain profile matter
FLAT, ARMS, WAIST, RAMP = "G1-Walk-Flat-Inspire-Play-v0", "G1-Walk-Flat-Inspire-Arms-Play-v0", "G1-Walk-Flat-Inspire-Waist-Play-v0", "G1-Walk-Flat-Inspire-StandRamp-Play-v0"
REGISTRY = [
    dict(id="v1", label="v1  stand (original plant)", task=FLAT, kind="stand", gains=True,
         ckpts=["logs/lab_checkpoints/v1/model_1499_rslrl5.pt", "logs/rsl_rl/g1_walk_flat_inspire/2026-10-28_20-48-49/model_1499.pt"]),
    dict(id="v2", label="v2  walk, legs only (original plant)", task=FLAT, kind="walk", gains=True, ckpts=["logs/lab_checkpoints/v2/model_2999.pt"]),
    dict(id="v3", label="v3  walk + waist (original plant)", task=WAIST, kind="walk", gains=True, ckpts=["logs/lab_checkpoints/v3/model_2999.pt"]),
    dict(id="v4", label="v4  walk, arms free (original plant)", task=ARMS, kind="walk", gains=True, ckpts=["logs/lab_checkpoints/v4/model_2999.pt"]),
    dict(id="v6", label="v6  walk + stand ramp (original plant)", task=RAMP, kind="walk", gains=True, ckpts=["logs/lab_checkpoints/v6/model_3498.pt"]),
    dict(id="v7", label="v7  stand (real plant)", task="G1-Walk-Flat-Inspire-V7-Play-v0", kind="stand", ckpts=["logs/lab_checkpoints/v7/model_1499.pt"]),
    dict(id="v8", label="v8  walk (real plant)", task="G1-Walk-Flat-Inspire-V8-Play-v0", kind="walk", ckpts=["logs/lab_checkpoints/v8/model_2999.pt"]),
    dict(id="v9a", label="v9a  walk, base height", task="G1-Walk-Flat-Inspire-V9a-Play-v0", kind="walk", ckpts=["logs/lab_checkpoints/v9a/model_4498.pt"]),
    dict(id="v9b", label="v9b  walk, base height + pose", task="G1-Walk-Flat-Inspire-V9b-Play-v0", kind="walk", ckpts=["logs/lab_checkpoints/v9b/model_2999.pt"]),
    dict(id="v9c", label="v9c  walk, v9b without the deadband", task="G1-Walk-Flat-Inspire-V9c-Play-v0", kind="walk", ckpts=["logs/lab_checkpoints/v9c/model_4498.pt"]),
    dict(id="v9d", label="v9d  walk, sagittal pose (stopped at 3850)", task="G1-Walk-Flat-Inspire-V9d-Play-v0", kind="walk", ckpts=["logs/lab_checkpoints/v9d/model_3850.pt"]),
    dict(id="v9e", label="v9e  walk, stand-only pose pull", task="G1-Walk-Flat-Inspire-V9e-Play-v0", kind="walk", ckpts=["logs/lab_checkpoints/v9e/model_4498.pt"]),
    dict(id="v11", label="v11  stand + turn in place (slow speeds)", task="G1-Walk-Flat-Inspire-V11-Play-v0", kind="stand-turn",
         ckpts=["logs/lab_checkpoints/v11/model_5998.pt"]),
    dict(id="v10", label="v10  stand (real plant, posture fix)", task="G1-Walk-Flat-Inspire-V10-Play-v0", kind="stand",
         ckpts=["logs/lab_checkpoints/v10/model_1499.pt"]),
]
# when this tree is the lab's, the runs live in logs/rsl_rl/<experiment>/<run>/model_*.pt
LAB_RUNS = {"v2": None, "v7": "g1_v7_stand_real/2026-10-30_00-27-55/model_1499.pt", "v8": "g1_v8_walk_real/2026-10-30_00-26-40/model_2999.pt",
            "v9a": "g1_v9a_walk_height/2026-10-31_12-18-48/model_4498.pt", "v9b": "g1_v9b_walk_pose/2026-10-30_08-58-24/model_2999.pt",
            "v9c": "g1_v9c_walk_slow/2026-10-31_10-20-46/model_4498.pt", "v9d": "g1_v9d_walk_sagittal/2026-10-31_12-18-48/model_3850.pt",
            "v9e": "g1_v9e_walk_standpull/2026-10-31_13-37-16/model_4498.pt",
            "v11": "g1_v11_stand_turn/*/model_5998.pt", "v10": "g1_v10_stand_pose/2026-10-30_08-59-54/model_1499.pt",
            "v4": "g1_walk_flat_inspire_arms/2026-10-29_03-22-08/model_2999.pt"}


def find_checkpoint(entry):
    cands = list(entry["ckpts"])
    if LAB_RUNS.get(entry["id"]):
        cands.append("logs/rsl_rl/" + LAB_RUNS[entry["id"]])
    for c in cands:
        for p in sorted(glob.glob(os.path.join(REPO_ROOT, c))):
            if os.path.isfile(p):
                return p
    return None


class Manager:
    """Owns the worker process; the only place that starts and stops it."""

    def __init__(self, worker_port, usd_path):
        self.worker_port = worker_port
        self.usd_path = usd_path
        self.proc = None
        self.lock = threading.Lock()
        self.status, self.error, self.current, self.gains = "idle", "", None, "sim"
        self.saved = {"target": {}, "config": {}, "command": {}}  # replayed into each new worker
        self.generation = 0
        self.log_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "play_policy_worker.log")

    def _stop(self):
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()  # Isaac sometimes ignores SIGTERM; this is our own child, so it is safe
                p.wait()

    def select(self, pid, gains):
        entry = next((e for e in REGISTRY if e["id"] == pid), None)
        ckpt = find_checkpoint(entry) if entry else None
        if not entry or not ckpt:
            return False, f"no checkpoint for {pid} on this machine"
        with self.lock:
            self.generation += 1
            gen = self.generation
            self.status, self.error, self.current, self.gains = "loading", "", pid, gains
        threading.Thread(target=self._run, args=(gen, entry, ckpt, gains), daemon=True).start()
        return True, ""

    def _run(self, gen, entry, ckpt, gains):
        with self.lock:
            self._stop()
        env = dict(os.environ)
        env["G1_USD_PATH"] = self.usd_path
        env["G1_GAIN_PROFILE"] = gains if entry.get("gains") else "sim"
        cmd = [sys.executable, "-u", WORKER, "--task", entry["task"], "--checkpoint", ckpt, "--label", entry["id"],
               "--port", str(self.worker_port)]
        log = open(self.log_path, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        with self.lock:
            if gen != self.generation:  # another selection arrived while we were starting
                proc.terminate()
                return
            self.proc = proc
        deadline = time.time() + 240
        while time.time() < deadline:
            time.sleep(1.0)
            with self.lock:
                if gen != self.generation:
                    return
            if proc.poll() is not None:
                tail = open(self.log_path, errors="replace").read()[-1500:]
                with self.lock:
                    self.status, self.error = "error", f"worker exited ({proc.returncode}). Log tail:\n{tail}"
                return
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.worker_port}/state", timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, OSError):
                continue
        else:
            with self.lock:
                self.status, self.error = "error", "worker did not become ready in 240 s"
            return
        for path, key in (("/target", "target"), ("/config", "config"), ("/command", "command")):  # carry the page state over
            body = {"values": self.saved["target"]} if key == "target" else self.saved[key]
            if body and body != {"values": {}}:
                self.forward("POST", path, json.dumps(body).encode(), force=True)
        with self.lock:
            if gen == self.generation:
                self.status = "ready"

    def forward(self, method, path, data=None, force=False):
        if self.status != "ready" and not force:
            return 503, b'{"loading": true}', "application/json"
        req = urllib.request.Request(f"http://127.0.0.1:{self.worker_port}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read(), r.headers.get("Content-Type", "application/json")
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            return 502, json.dumps({"error": str(exc)}).encode(), "application/json"

    def remember(self, path, body):
        if path == "/target":
            self.saved["target"].update(body.get("values", {}))
        elif path == "/config":
            self.saved["config"].update(body)
        elif path == "/command":
            self.saved["command"].update(body)

    def shutdown(self):
        with self.lock:
            self.generation += 1
            self._stop()


manager = None
args = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/policies":
            return self._send(200, json.dumps([{"id": e["id"], "label": e["label"], "kind": e["kind"], "gains": bool(e.get("gains")),
                                                "available": find_checkpoint(e) is not None} for e in REGISTRY]))
        if self.path == "/status":
            return self._send(200, json.dumps({"status": manager.status, "current": manager.current, "gains": manager.gains,
                                               "error": manager.error}))
        if self.path.startswith("/stream"):
            if manager.status != "ready":
                return self._send(503, "loading", "text/plain")
            try:
                conn = http.client.HTTPConnection("127.0.0.1", manager.worker_port, timeout=5)
                conn.request("GET", "/stream")
                resp = conn.getresponse()
            except OSError:
                return self._send(502, "worker unreachable", "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", resp.getheader("Content-Type", "multipart/x-mixed-replace; boundary=frame"))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                conn.close()
            return
        if self.path == "/state":
            code, body, ctype = manager.forward("GET", "/state")
            return self._send(code, body, ctype)
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) or b"{}"
        try:
            body = json.loads(raw)
        except ValueError:
            return self._send(400, "bad json", "text/plain")
        if self.path == "/policy":
            ok, msg = manager.select(body.get("id"), body.get("gains", "sim") if body.get("gains") in ("sim", "unitree") else "sim")
            return self._send(200 if ok else 400, json.dumps({"ok": ok, "error": msg}))
        if self.path in ("/target", "/config", "/command"):
            manager.remember(self.path, body)
        code, out, ctype = manager.forward("POST", self.path, raw)
        self._send(code, out, ctype)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 policy playground</title>
<style>
:root{--bg:#0f1318;--panel:#171d25;--line:#2a3340;--text:#e6ebf2;--mute:#8a96a8;--acc:#3ea6ff;--good:#3ecf8e;--warn:#ffb84d;--bad:#ff5d5d}
@media (prefers-color-scheme: light){:root{--bg:#f3f5f8;--panel:#fff;--line:#d5dbe4;--text:#18202b;--mute:#5d6b7e;--acc:#0b6bcb;--good:#12915b;--warn:#b86e00;--bad:#c62828}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0 8px 0 0}
select{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit;max-width:100%}
.pill{padding:3px 10px;border-radius:999px;border:1px solid var(--line);font-size:12px}.pill.ready{color:var(--good);border-color:var(--good)}
.pill.loading{color:var(--warn);border-color:var(--warn)}.pill.error{color:var(--bad);border-color:var(--bad)}
main{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(330px,1fr);gap:14px;padding:14px;max-width:1500px;margin:auto}
@media(max-width:980px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:12px}.panel>:first-child{margin-top:0}
.view img{width:100%;border-radius:6px;display:block;background:#000;aspect-ratio:16/10;object-fit:contain}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));gap:8px;margin-top:10px}
.card{border:1px solid var(--line);border-radius:6px;padding:6px 8px}.card b{display:block;font-size:17px}.card small{color:var(--mute)}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}
button{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px 10px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)}button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
h2{font-size:13px;margin:0 0 6px;color:var(--mute);text-transform:uppercase;letter-spacing:.05em}
.j{display:grid;grid-template-columns:120px 1fr 120px;gap:8px;align-items:center;margin:3px 0}
.j label{color:var(--mute);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.j output{font-variant-numeric:tabular-nums;text-align:right;font-size:12px}
input[type=range]{width:100%;accent-color:var(--acc);height:18px}
.out{color:var(--warn)}label.chk{display:flex;gap:6px;align-items:center}.note{color:var(--mute);font-size:12px}
details summary{cursor:pointer;color:var(--mute);text-transform:uppercase;letter-spacing:.05em;font-size:13px;margin-bottom:6px}
pre{white-space:pre-wrap;color:var(--bad);font-size:12px;margin:6px 0 0}
</style></head><body>
<header><h1>Unitree G1 policy playground</h1>
  <select id="policy"></select>
  <label class="chk" id="gainsbox" style="display:none">plant gains <select id="gains"><option value="sim">sim (original)</option><option value="unitree">unitree (real)</option></select></label>
  <span class="pill" id="status">idle</span><span class="note" id="kind"></span>
</header>
<main>
<section>
 <div class="panel view">
  <img id="cam" alt="camera">
  <div class="cards">
    <div class="card"><small>Base height</small><b id="t_z">-</b></div>
    <div class="card"><small>Tilt</small><b id="t_tilt">-</b></div>
    <div class="card"><small>Command vx / vy / wz</small><b id="t_cmd">-</b></div>
    <div class="card"><small>Achieved vx / vy / wz</small><b id="t_ach">-</b></div>
    <div class="card"><small>Distance from start</small><b id="t_dist">-</b></div>
    <div class="card"><small>Foot lifts</small><b id="t_lifts">-</b></div>
    <div class="card"><small>Falls</small><b id="t_falls">-</b></div>
  </div>
  <div class="row"><span class="note">Camera</span>
    <button data-cam="front">front</button><button data-cam="side">side</button><button data-cam="back">back</button><button data-cam="close">close</button><button data-cam="far">far</button>
    <span class="note" style="margin-left:auto" id="t_fps"></span></div>
  <div class="row"><span class="note">Disturb</span>
    <button id="push3">shove 0.3 m/s</button><button id="push5">shove 0.5 m/s</button><button id="reset">reset robot</button></div>
  <pre id="err"></pre>
 </div>
</section>
<section>
 <div class="panel">
  <h2>Velocity command</h2>
  <div id="cmd"></div>
  <div class="row"><button id="stop">stop (zero)</button>
    <button data-p="0.2,0,0">creep 0.2</button><button data-p="0.5,0,0">forward 0.5</button><button data-p="1.0,0,0">forward 1.0</button>
    <button data-p="0,0.3,0">left 0.3</button><button data-p="0,-0.3,0">right 0.3</button>
    <button data-p="0,0,0.5">turn left 0.5</button><button data-p="0,0,-0.5">turn right 0.5</button></div>
  <div class="row">
    <label class="chk"><input type="checkbox" id="ramp"> ramp at 1 m/s&sup2; / 2 rad/s&sup2; (like the controller)</label>
    <label class="chk"><input type="checkbox" id="deadband"> 0.05 deadband</label>
  </div>
  <div class="row"><label class="chk"><input type="checkbox" id="keys" checked> keyboard drive: W/S forward-back, A/D left-right, Q/E turn, space stops</label>
    <label class="chk">key speed <input type="range" id="kspeed" min="0.1" max="1.0" step="0.05" value="0.4" style="width:100px"><output id="kspeed_o">0.40</output></label></div>
  <p class="note">The shaded part of each slider is the range the walking policies were trained on. Stand-only policies (v1, v7, v10) see the command but were not trained to walk.</p>
 </div>
 <details class="panel" open><summary>Arms</summary>
  <div class="row">
    <button id="lock" class="on" title="locked: targets are clamped to the safe range. unlocked: any pose within the joint limits">safe range: LOCKED</button>
    <label class="chk">arm speed <input type="range" id="speed" min="0.2" max="4" step="0.1" value="1" style="width:100px"><output id="speed_o">1.0</output> rad/s</label>
  </div>
  <div class="row">
    <button id="p_rest">rest pose</button><button id="p_pitch_p">pitch +</button><button id="p_pitch_n">pitch &minus;</button>
    <button id="p_roll">arms out</button><button id="p_copy">copy left &rarr; right (mirrored)</button></div>
  <div class="row"><button id="random">random reach (human-like)</button>
    <label class="chk">speed <input type="range" id="rspeed" min="0.3" max="3" step="0.1" value="1" style="width:100px"><output id="rspeed_o">1.0</output> rad/s</label></div>
  <p class="note">Green band = safe range (60% of the training arm range around the default). UNLOCK to reach any pose: outside the band the robot
  drifts and steps, far outside it falls (then respawns).</p>
  <div id="sliders"></div>
 </details>
</section>
</main>
<script>
const $=id=>document.getElementById(id);
const post=(url,body)=>fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).catch(()=>{});
let joints=[], pendingArm={}, armTimer=null, last=null, ready=false, policies=[];
const KIND={stand:'stand-only policy',walk:'walking policy','stand-turn':'stand + turn-in-place policy (slow speeds)'};
// ---------------- policy selection
async function loadPolicies(){
  policies=await (await fetch('/policies')).json(); const sel=$('policy'); sel.innerHTML='';
  for(const p of policies){const o=document.createElement('option'); o.value=p.id; o.textContent=p.label+(p.available?'':'  (no checkpoint here)'); o.disabled=!p.available; sel.appendChild(o)}
}
function choose(){const id=$('policy').value; const p=policies.find(x=>x.id===id); $('gainsbox').style.display=p&&p.gains?'flex':'none';
  $('kind').textContent=p?(KIND[p.kind]):''; post('/policy',{id,gains:$('gains').value}); setStatus('loading'); }
$('policy').onchange=choose; $('gains').onchange=choose;
function setStatus(s,err){const e=$('status'); e.className='pill '+s; e.textContent=s==='loading'?'loading policy (about a minute)...':s; $('err').textContent=err||'';
  if(s==='ready'&&!ready){ready=true; $('cam').src='/stream?'+Date.now(); joints=[]; }
  if(s!=='ready'){ready=false; $('cam').removeAttribute('src')}}
// ---------------- velocity command
const AXES=[{k:'vx',lab:'forward vx',lo:-1,hi:1.5,tlo:-0.4,thi:1.0},{k:'vy',lab:'sideways vy',lo:-1,hi:1,tlo:-0.5,thi:0.5},{k:'wz',lab:'yaw rate wz',lo:-2,hi:2,tlo:-1,thi:1}];
function buildCmd(){const box=$('cmd'); for(const a of AXES){const r=document.createElement('div'); r.className='j'; r.style.gridTemplateColumns='110px 1fr 70px';
  const l=document.createElement('label'); l.textContent=a.lab; const i=document.createElement('input'); i.type='range'; i.min=a.lo; i.max=a.hi; i.step=0.01; i.value=0; i.id='c_'+a.k;
  const p=(a.tlo-a.lo)/(a.hi-a.lo)*100, q=(a.thi-a.lo)/(a.hi-a.lo)*100; i.style.background=`linear-gradient(to right,transparent ${p}%,rgba(62,207,142,.35) ${p}%,rgba(62,207,142,.35) ${q}%,transparent ${q}%)`;
  const o=document.createElement('output'); o.id='co_'+a.k; o.textContent='0.00'; i.oninput=()=>{o.textContent=(+i.value).toFixed(2); sendCmd()}; r.append(l,i,o); box.appendChild(r)}}
function sendCmd(){post('/command',{vx:+$('c_vx').value,vy:+$('c_vy').value,wz:+$('c_wz').value})}
function setCmd(vx,vy,wz){for(const [k,v] of [['vx',vx],['vy',vy],['wz',wz]]){$('c_'+k).value=v; $('co_'+k).textContent=(+v).toFixed(2)} sendCmd()}
$('stop').onclick=()=>setCmd(0,0,0);
document.querySelectorAll('[data-p]').forEach(b=>b.onclick=()=>setCmd(...b.dataset.p.split(',').map(Number)));
$('ramp').onchange=e=>post('/config',{ramp:e.target.checked}); $('deadband').onchange=e=>post('/config',{deadband:e.target.checked});
$('kspeed').oninput=e=>$('kspeed_o').textContent=(+e.target.value).toFixed(2);
const KEYS={w:['vx',1],s:['vx',-1],a:['vy',1],d:['vy',-1],q:['wz',1],e:['wz',-1]}, down=new Set();
function keyCmd(){const s=+$('kspeed').value; const v={vx:0,vy:0,wz:0}; for(const k of down){const [ax,sg]=KEYS[k]; v[ax]+=sg*(ax==='wz'?s*1.5:s)}; setCmd(v.vx,v.vy,v.wz)}
addEventListener('keydown',e=>{if(!$('keys').checked||e.repeat||/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)&&document.activeElement.type!=='range')return;
  const k=e.key.toLowerCase(); if(k===' '){e.preventDefault(); down.clear(); setCmd(0,0,0)} else if(KEYS[k]){down.add(k); keyCmd()}});
addEventListener('keyup',e=>{const k=e.key.toLowerCase(); if(down.delete(k)) keyCmd()});
// ---------------- arms
function buildArms(){const box=$('sliders'); box.innerHTML='';
  for(const side of ['left','right']){const h=document.createElement('div'); h.className='note'; h.style.margin='8px 0 2px'; h.textContent=side+' arm'; box.appendChild(h);
    for(const j of joints.filter(j=>j.name.startsWith(side))){
      const row=document.createElement('div'); row.className='j'; row.style.gridTemplateColumns='120px 1fr 60px';
      const lab=document.createElement('label'); lab.textContent=j.name.replace(side+'_','').replace('_joint',''); lab.title=j.name;
      const inp=document.createElement('input'); inp.type='range'; inp.min=j.lo; inp.max=j.hi; inp.step=0.01; inp.value=j.target; inp.id='s_'+j.name;
      const a=(j.rec_lo-j.lo)/(j.hi-j.lo)*100, b=(j.rec_hi-j.lo)/(j.hi-j.lo)*100;
      inp.style.background=`linear-gradient(to right,transparent ${a}%,rgba(62,207,142,.35) ${a}%,rgba(62,207,142,.35) ${b}%,transparent ${b}%)`;
      const out=document.createElement('output'); out.id='o_'+j.name; out.textContent=(+j.target).toFixed(2);
      inp.oninput=()=>{out.textContent=(+inp.value).toFixed(2); out.className=(+inp.value<j.rec_lo-1e-6||+inp.value>j.rec_hi+1e-6)?'out':''; pendingArm[j.name]=+inp.value; if(!armTimer) armTimer=setTimeout(()=>{armTimer=null; const v=pendingArm; pendingArm={}; post('/target',{values:v})},30)};
      row.append(lab,inp,out); box.appendChild(row)}}}
function setArms(values){for(const [n,v] of Object.entries(values)){const i=$('s_'+n); if(i){i.value=v; i.dispatchEvent(new Event('input'))}}}
function renderLock(on){const b=$('lock'); b.classList.toggle('on',on); b.textContent=on?'safe range: LOCKED':'safe range: UNLOCKED (robot may fall)'; b.style.background=on?'':'var(--bad)'; b.style.borderColor=on?'':'var(--bad)'; b.style.color=on?'':'#fff'}
async function setLock(on){await post('/config',{clamp:on}); const s=await (await fetch('/state')).json(); last=s;
  for(const j of s.joints){const i=$('s_'+j.name); if(i){i.value=j.target; const o=$('o_'+j.name); o.textContent=(+j.target).toFixed(2); o.className=(j.target<j.rec_lo-1e-6||j.target>j.rec_hi+1e-6)?'out':''}} renderLock(on)}
$('lock').onclick=()=>setLock(!(last?last.clamp:true));
$('speed').oninput=e=>{$('speed_o').textContent=(+e.target.value).toFixed(1); post('/config',{speed:+e.target.value})};
$('rspeed').oninput=e=>{$('rspeed_o').textContent=(+e.target.value).toFixed(1); post('/config',{random_speed:+e.target.value})};
$('random').onclick=()=>post('/config',{random_reach:!(last&&last.random_reach)});
const P=f=>{const o={}; for(const j of joints) o[j.name]=f(j); return o}, nr=()=>post('/config',{random_reach:false});
$('p_rest').onclick=()=>{nr(); setArms(P(j=>j.default))};
$('p_pitch_p').onclick=()=>{nr(); setArms(P(j=>j.name.endsWith('shoulder_pitch_joint')?j.default+0.6*(j.rec_hi-j.default):j.default))};
$('p_pitch_n').onclick=()=>{nr(); setArms(P(j=>j.name.endsWith('shoulder_pitch_joint')?j.default+0.6*(j.rec_lo-j.default):j.default))};
$('p_roll').onclick=()=>{nr(); setArms(P(j=>j.name.endsWith('shoulder_roll_joint')?(j.name.startsWith('left')?j.rec_hi:j.rec_lo):j.default))};
$('p_copy').onclick=()=>{const o={}; for(const j of joints.filter(j=>j.name.startsWith('left'))){const r=joints.find(k=>k.name==='right'+j.name.slice(4)); const f=/roll|yaw/.test(j.name)?-1:1; o[r.name]=r.default+f*(+$('s_'+j.name).value-j.default)} setArms(o)};
// ---------------- disturb / camera
$('reset').onclick=()=>post('/action',{name:'reset'}); $('push3').onclick=()=>post('/action',{name:'push',mag:0.3}); $('push5').onclick=()=>post('/action',{name:'push',mag:0.5});
document.querySelectorAll('[data-cam]').forEach(b=>b.onclick=()=>post('/action',{name:'camera',view:b.dataset.cam}));
// ---------------- polling
async function poll(){
  try{
    const st=await (await fetch('/status')).json();
    if(st.current&&$('policy').value!==st.current&&document.activeElement!==$('policy')) $('policy').value=st.current;
    if(st.gains) $('gains').value=st.gains;
    const p=policies.find(x=>x.id===st.current); $('gainsbox').style.display=p&&p.gains?'flex':'none'; $('kind').textContent=p?(KIND[p.kind]):'';
    setStatus(st.status,st.error);
    if(st.status==='ready'){
      const r=await fetch('/state'); if(r.ok){const s=await r.json(); last=s;
        if(!joints.length){joints=s.joints; buildArms()}
        const t=s.telemetry, c=s.command; if(t.z!==undefined){
        $('t_z').textContent=t.z.toFixed(3)+' m'; $('t_tilt').textContent=t.tilt.toFixed(1)+'°';
        $('t_cmd').textContent=c.applied.map(x=>x.toFixed(2)).join(' / '); $('t_ach').textContent=[t.vx,t.vy,t.wz].map(x=>x.toFixed(2)).join(' / ');
        $('t_dist').textContent=(t.dist_cm/100).toFixed(2)+' m'; $('t_lifts').textContent=t.lifts; $('t_falls').textContent=t.falls; $('t_fps').textContent=t.fps.toFixed(0)+' sim steps/s'}
        $('random').classList.toggle('on',s.random_reach); renderLock(s.clamp); $('ramp').checked=c.ramp; $('deadband').checked=c.deadband;
        if(s.random_reach&&document.activeElement.type!=='range') for(const j of s.joints){const i=$('s_'+j.name); if(i){i.value=j.target; $('o_'+j.name).textContent=(+j.target).toFixed(2)}}
      }
    }
  }catch(e){}
  setTimeout(poll,250);
}
buildCmd(); loadPolicies().then(()=>{poll()});
</script></body></html>
"""


def main():
    global manager, args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from other machines (Tailscale)")
    ap.add_argument("--worker-port", type=int, default=18765)
    ap.add_argument("--policy", default="v10", help="policy loaded at start")
    ap.add_argument("--gains", choices=["sim", "unitree"], default="sim")
    ap.add_argument("--usd", default=os.environ.get("G1_USD_PATH", ""), help="G1 USD asset (G1_USD_PATH)")
    args = ap.parse_args()
    if not args.usd:
        cands = glob.glob(os.path.expanduser("~/Programming/*/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd"))
        cands += glob.glob(os.path.expanduser("~/Programming/*/*/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd"))
        args.usd = cands[0] if cands else ""
    manager = Manager(args.worker_port, args.usd)
    atexit.register(manager.shutdown)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: (manager.shutdown(), os._exit(0)))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    ok, msg = manager.select(args.policy, args.gains)
    print(f"[play_policy] http://{args.host}:{args.port}   initial policy {args.policy}: {'loading' if ok else msg}", flush=True)
    print(f"[play_policy] worker log: {manager.log_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
