#!/usr/bin/env python3
"""Move the arms freely from a web page while the standing policy (v10) keeps the robot upright.

The simulation runs here (headless, with a camera); a small web server streams the camera as MJPEG and
takes slider values for the 14 arm joints. Nothing but the Python standard library is needed for the web part.

    python scripts/play_arms.py --checkpoint logs/rsl_rl/g1_v10_stand_pose/<run>/model_1499.pt   [--port 8765]

then open  http://<this machine>:8765  (on the lab desktop over Tailscale:  http://100.116.200.54:8765 , or tunnel:
ssh -L 8765:localhost:8765 cesar@100.116.200.54  and open  http://localhost:8765 ).

The velocity command is fixed at zero: the policy only has to stand. The arms are not controlled by the policy; each step this
script sets their position targets (rate-limited so they sweep instead of teleporting), exactly like scripts/evaluate_standing.py.

Page: live view, telemetry (base height, tilt, drift, foot lifts, falls), one slider per arm joint, presets, "limit to the
recommended range" (60% of the training arm range around the default pose: what v10 handles best), a random-reach mode
(human-imitation style), mirrored copy left->right, shoves, reset and camera views.
"""
import argparse
import io
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-Walk-Flat-Inspire-V10-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=600)
parser.add_argument("--stream-every", type=int, default=2, help="render one frame every N control steps (50 Hz / N)")
parser.add_argument("--jpeg-quality", type=int, default=80)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app = AppLauncher(args).app

import gymnasium as gym
import torch
from PIL import Image
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import g1_rl  # noqa: F401
from g1_rl.body_env_cfg import ARM_POSE_RANGES
from g1_rl.robot_cfg import ARM_JOINT_NAMES

RECOMMENDED_FRACTION = 0.6  # of the training arm range around the default pose (the "mimic" scenarios of evaluate_standing.py)
CONTACT_N = 10.0
MIN_AIR_S = 0.1
CAMERAS = {  # eye, lookat: relative to the robot's pelvis
    "front": ((2.2, -1.1, 0.45), (0.0, 0.0, 0.1)),
    "side": ((0.3, -2.8, 0.4), (0.0, 0.0, 0.1)),
    "back": ((-2.2, -1.1, 0.5), (0.0, 0.0, 0.1)),
    "close": ((1.4, -0.7, 0.55), (0.0, 0.0, 0.2)),
}


class Shared:
    """State shared between the simulation loop (main thread) and the HTTP handlers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame_cond = threading.Condition()
        self.jpeg = b""
        self.frame_id = 0
        self.target = {}  # joint name -> absolute target (rad)
        self.current = {}
        self.speed = 1.0  # rad/s, arm rate limit
        self.clamp = True
        self.random_reach = False
        self.random_speed = 1.0
        self.requests = []  # ("reset",) ("push", mag) ("camera", name)
        self.telemetry = {}
        self.joints = []  # static info for the page


shared = Shared()


def encode_jpeg(frame):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=args.jpeg_quality)
    return buf.getvalue()


# ----------------------------------------------------------------------------- web
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 arms | v10 standing</title>
<style>
:root{--bg:#0f1318;--panel:#171d25;--line:#2a3340;--text:#e6ebf2;--mute:#8a96a8;--acc:#3ea6ff;--good:#3ecf8e;--warn:#ffb84d;--bad:#ff5d5d}
@media (prefers-color-scheme: light){:root{--bg:#f3f5f8;--panel:#fff;--line:#d5dbe4;--text:#18202b;--mute:#5d6b7e;--acc:#0b6bcb;--good:#12915b;--warn:#b86e00;--bad:#c62828}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:16px;margin:0}header span{color:var(--mute)}
main{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(320px,1fr);gap:14px;padding:14px;max-width:1500px;margin:auto}
@media(max-width:980px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.view img{width:100%;border-radius:6px;display:block;background:#000;aspect-ratio:16/10;object-fit:contain}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:8px;margin-top:10px}
.card{border:1px solid var(--line);border-radius:6px;padding:6px 8px}.card b{display:block;font-size:18px}.card small{color:var(--mute)}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}
button{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px 10px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)}button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
h2{font-size:13px;margin:12px 0 4px;color:var(--mute);text-transform:uppercase;letter-spacing:.05em}
.j{display:grid;grid-template-columns:130px 1fr 74px;gap:8px;align-items:center;margin:3px 0}
.j label{color:var(--mute);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.j output{font-variant-numeric:tabular-nums;text-align:right;font-size:12px}
input[type=range]{width:100%;accent-color:var(--acc);height:18px}
.out{color:var(--warn)}label.chk{display:flex;gap:6px;align-items:center}
.note{color:var(--mute);font-size:12px}
</style></head><body>
<header><h1>Unitree G1 | v10 standing policy</h1><span>zero velocity command &middot; move the arms, the policy keeps the robot upright</span></header>
<main>
<section class="panel view">
  <img id="cam" src="/stream" alt="camera">
  <div class="cards">
    <div class="card"><small>Base height</small><b id="t_z">-</b></div>
    <div class="card"><small>Tilt</small><b id="t_tilt">-</b></div>
    <div class="card"><small>Drift from start</small><b id="t_drift">-</b></div>
    <div class="card"><small>Base speed</small><b id="t_speed">-</b></div>
    <div class="card"><small>Foot lifts</small><b id="t_lifts">-</b></div>
    <div class="card"><small>Falls</small><b id="t_falls">-</b></div>
  </div>
  <div class="row"><span class="note">Camera</span>
    <button data-cam="front">front</button><button data-cam="side">side</button><button data-cam="back">back</button><button data-cam="close">close</button>
    <span class="note" style="margin-left:auto" id="t_fps"></span></div>
  <div class="row"><span class="note">Disturb</span>
    <button id="push3">shove 0.3 m/s</button><button id="push5">shove 0.5 m/s</button><button id="reset">reset robot</button></div>
</section>
<section class="panel">
  <div class="row">
    <button id="lock" class="on" title="locked: targets are clamped to the recommended range. unlocked: any pose within the joint limits">safe range: LOCKED</button>
    <label class="chk">arm speed <input type="range" id="speed" min="0.2" max="4" step="0.1" value="1" style="width:110px"><output id="speed_o">1.0</output> rad/s</label>
  </div>
  <div class="row">
    <button id="p_rest">rest pose</button><button id="p_pitch_p">pitch +</button><button id="p_pitch_n">pitch &minus;</button>
    <button id="p_roll">arms out</button><button id="p_copy">copy left &rarr; right (mirrored)</button>
  </div>
  <div class="row">
    <button id="random">random reach (human-like)</button>
    <label class="chk">speed <input type="range" id="rspeed" min="0.3" max="3" step="0.1" value="1" style="width:100px"><output id="rspeed_o">1.0</output> rad/s</label>
  </div>
  <p class="note">Green band on each slider = safe range (60% of the training arm range around the default). While LOCKED the targets are held
  inside it. UNLOCK to reach any pose the joints allow: outside the band the robot drifts and steps, far outside it falls (v10 fails with the arms
  held far to one side). After a fall it respawns upright and the arms keep going to your slider values; "reset robot" also re-zeroes the drift.</p>
  <div id="sliders"></div>
</section>
</main>
<script>
const $=id=>document.getElementById(id);
let joints=[], sending=null, pending={}, lastState=null;
function post(url,body){return fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).catch(()=>{})}
function build(){
  const box=$('sliders'); box.innerHTML='';
  for(const side of ['left','right']){
    const h=document.createElement('h2'); h.textContent=side+' arm'; box.appendChild(h);
    for(const j of joints.filter(j=>j.name.startsWith(side))){
      const row=document.createElement('div'); row.className='j';
      const lab=document.createElement('label'); lab.textContent=j.name.replace(side+'_','').replace('_joint',''); lab.title=j.name;
      const inp=document.createElement('input'); inp.type='range'; inp.min=j.lo; inp.max=j.hi; inp.step=0.01; inp.value=j.target; inp.id='s_'+j.name;
      const a=(j.rec_lo-j.lo)/(j.hi-j.lo)*100, b=(j.rec_hi-j.lo)/(j.hi-j.lo)*100;
      inp.style.background=`linear-gradient(to right,transparent ${a}%,rgba(62,207,142,.35) ${a}%,rgba(62,207,142,.35) ${b}%,transparent ${b}%)`;
      const out=document.createElement('output'); out.id='o_'+j.name; out.textContent=(+j.target).toFixed(2);
      inp.addEventListener('input',()=>{out.textContent=(+inp.value).toFixed(2); out.className=(+inp.value<j.rec_lo-1e-6||+inp.value>j.rec_hi+1e-6)?'out':''; pending[j.name]=+inp.value; if(!sending) sending=setTimeout(flush,30)});
      row.append(lab,inp,out); box.appendChild(row);
    }
  }
}
function flush(){sending=null; const v=pending; pending={}; post('/target',{values:v})}
function setAll(values){ // values: name->rad, from presets
  for(const [n,v] of Object.entries(values)){const i=$('s_'+n); if(i){i.value=v; i.dispatchEvent(new Event('input'))}}
}
async function poll(){
  try{
    const s=await (await fetch('/state')).json(); lastState=s;
    if(!joints.length){joints=s.joints; build()}
    const t=s.telemetry;
    $('t_z').textContent=t.z.toFixed(3)+' m'; $('t_tilt').textContent=t.tilt.toFixed(1)+'°';
    $('t_drift').textContent=t.drift_cm.toFixed(1)+' cm'; $('t_speed').textContent=(t.speed*100).toFixed(1)+' cm/s';
    $('t_lifts').textContent=t.lifts; $('t_falls').textContent=t.falls; $('t_fps').textContent=t.fps.toFixed(0)+' sim steps/s';
    $('random').classList.toggle('on',s.random_reach); renderLock(s.clamp);
    if(document.activeElement.type!=='range'){ // keep sliders in sync when the server changes them (presets, random reach)
      for(const j of s.joints){const i=$('s_'+j.name); if(i&&s.random_reach){i.value=j.target; $('o_'+j.name).textContent=(+j.target).toFixed(2)}}
    }
  }catch(e){}
  setTimeout(poll,250);
}
$('speed').oninput=e=>{$('speed_o').textContent=(+e.target.value).toFixed(1); post('/config',{speed:+e.target.value})};
$('rspeed').oninput=e=>{$('rspeed_o').textContent=(+e.target.value).toFixed(1); post('/config',{random_speed:+e.target.value})};
async function setLock(on){
  await post('/config',{clamp:on});
  const s=await (await fetch('/state')).json(); lastState=s; // locking snaps targets back into the safe band: refresh the sliders
  for(const j of s.joints){const i=$('s_'+j.name); if(i){i.value=j.target; const o=$('o_'+j.name); o.textContent=(+j.target).toFixed(2); o.className=(j.target<j.rec_lo-1e-6||j.target>j.rec_hi+1e-6)?'out':''}}
  renderLock(on);
}
function renderLock(on){const b=$('lock'); b.classList.toggle('on',on); b.textContent=on?'safe range: LOCKED':'safe range: UNLOCKED (robot may fall)'; b.style.background=on?'':'var(--bad)'; b.style.borderColor=on?'':'var(--bad)'; b.style.color=on?'':'#fff'}
$('lock').onclick=()=>setLock(!(lastState?lastState.clamp:true));
$('random').onclick=()=>post('/config',{random_reach:!(lastState&&lastState.random_reach)});
$('reset').onclick=()=>post('/action',{name:'reset'});
$('push3').onclick=()=>post('/action',{name:'push',mag:0.3});
$('push5').onclick=()=>post('/action',{name:'push',mag:0.5});
document.querySelectorAll('[data-cam]').forEach(b=>b.onclick=()=>post('/action',{name:'camera',view:b.dataset.cam}));
const P=(f)=>{const o={}; for(const j of joints) o[j.name]=f(j); return o};
const rest=j=>j.default;
$('p_rest').onclick=()=>{post('/config',{random_reach:false}); setAll(P(rest))};
$('p_pitch_p').onclick=()=>{post('/config',{random_reach:false}); setAll(P(j=>j.name.endsWith('shoulder_pitch_joint')?j.default+0.6*(j.rec_hi-j.default):j.default))};
$('p_pitch_n').onclick=()=>{post('/config',{random_reach:false}); setAll(P(j=>j.name.endsWith('shoulder_pitch_joint')?j.default+0.6*(j.rec_lo-j.default):j.default))};
$('p_roll').onclick=()=>{post('/config',{random_reach:false}); setAll(P(j=>j.name.endsWith('shoulder_roll_joint')?(j.name.startsWith('left')?j.rec_hi:j.rec_lo):j.default))};
$('p_copy').onclick=()=>{const o={};for(const j of joints.filter(j=>j.name.startsWith('left'))){
  const r=joints.find(k=>k.name==='right'+j.name.slice(4)); const flip=/roll|yaw/.test(j.name)?-1:1;
  o[r.name]=r.default+flip*(+$('s_'+j.name).value-j.default)}; setAll(o)};
poll();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
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
        if self.path == "/state":
            with shared.lock:
                joints = [dict(j, target=shared.target[j["name"]], current=shared.current.get(j["name"], j["default"]))
                          for j in shared.joints]
                body = json.dumps({"joints": joints, "telemetry": shared.telemetry, "random_reach": shared.random_reach,
                                   "clamp": shared.clamp, "speed": shared.speed})
            return self._send(200, body)
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while True:
                    with shared.frame_cond:
                        shared.frame_cond.wait_for(lambda: shared.frame_id != last, timeout=2.0)
                        data, last = shared.jpeg, shared.frame_id
                    if not data:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send(400, "bad json", "text/plain")
        with shared.lock:
            if self.path == "/target":
                for name, v in body.get("values", {}).items():
                    if name in shared.target:
                        shared.target[name] = float(v)
                shared.random_reach = False
            elif self.path == "/config":
                if "speed" in body:
                    shared.speed = max(0.05, float(body["speed"]))
                if "random_speed" in body:
                    shared.random_speed = max(0.05, float(body["random_speed"]))
                if "clamp" in body:
                    shared.clamp = bool(body["clamp"])
                    if shared.clamp:  # re-locking pulls every stored target back into the safe band
                        for j in shared.joints:
                            shared.target[j["name"]] = min(max(shared.target[j["name"]], j["rec_lo"]), j["rec_hi"])
                if "random_reach" in body:
                    shared.random_reach = bool(body["random_reach"])
            elif self.path == "/action":
                shared.requests.append((body.get("name"), body))
            else:
                return self._send(404, "not found", "text/plain")
        self._send(200, "{}")


# ----------------------------------------------------------------------------- simulation
@torch.inference_mode()
def main():
    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    try:
        import importlib.metadata as _metadata

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))
    except ImportError:
        pass

    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    cmd_cfg.mixed_sampling = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.ranges.lin_vel_x = (0.0, 0.0)
    cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.episode_length_s = 1.0e6
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.commands, "arm_pose"):
        env_cfg.commands.arm_pose = None  # the arms are driven from the web page
    env_cfg.observations.policy.enable_corruption = False

    v = env_cfg.viewer
    v.origin_type, v.asset_name, v.env_index = "asset_root", "robot", 0
    v.eye, v.lookat = CAMERAS["front"]
    v.resolution = (args.width, args.height)

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.scene["robot"]
    dt = base.step_dt
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    default = robot.data.default_joint_pos[0, arm_ids].clone()
    soft = robot.data.soft_joint_pos_limits[0, arm_ids].clone()
    lo_full = torch.tensor([ARM_POSE_RANGES[n][0] for n in ARM_JOINT_NAMES], device=device)
    hi_full = torch.tensor([ARM_POSE_RANGES[n][1] for n in ARM_JOINT_NAMES], device=device)
    rec_lo = default + RECOMMENDED_FRACTION * (lo_full - default)
    rec_hi = default + RECOMMENDED_FRACTION * (hi_full - default)
    sensor = base.scene["contact_forces"]
    foot_ids = sensor.find_bodies(".*_ankle_roll_link")[0]

    with shared.lock:
        for i, n in enumerate(ARM_JOINT_NAMES):
            shared.target[n] = float(default[i])
            shared.joints.append({"name": n, "lo": float(soft[i, 0]), "hi": float(soft[i, 1]), "default": float(default[i]),
                                  "rec_lo": float(rec_lo[i]), "rec_hi": float(rec_hi[i])})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[play_arms] serving on http://{args.host}:{args.port}  (checkpoint {args.checkpoint})", flush=True)

    cur = default.clone()
    goal = default.clone()
    hold = 0.0
    obs = env.get_observations()
    start_xy = robot.data.root_pos_w[0, :2].clone()
    falls, lifts, air_run, t_wall, steps_done, fps = 0, 0, torch.zeros(2, device=device), time.time(), 0, 0.0
    step = 0
    next_t = time.time()
    while app.is_running():
        with shared.lock:
            reqs, shared.requests = shared.requests, []
            speed, clamp = shared.speed, shared.clamp
            random_reach, rspeed = shared.random_reach, shared.random_speed
            tgt = torch.tensor([shared.target[n] for n in ARM_JOINT_NAMES], device=device)
        for name, body in reqs:
            if name == "reset":
                env.reset()
                obs = env.get_observations()
                cur = default.clone()
                start_xy = robot.data.root_pos_w[0, :2].clone()
                falls, lifts = 0, 0
            elif name == "push":
                ang = float(torch.rand(1)) * 2 * math.pi
                vel = robot.data.root_vel_w.clone()
                vel[:, 0] += float(body.get("mag", 0.3)) * math.cos(ang)
                vel[:, 1] += float(body.get("mag", 0.3)) * math.sin(ang)
                robot.write_root_velocity_to_sim(vel)
            elif name == "camera" and body.get("view") in CAMERAS:
                eye, lookat = CAMERAS[body["view"]]
                try:
                    base.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
                except Exception as exc:  # noqa: BLE001
                    print("[play_arms] camera change failed:", exc, flush=True)

        if random_reach:  # human-imitation style reaches inside the recommended band
            if hold > 0:
                hold -= dt
            elif (goal - cur).abs().max() < 1e-3:
                goal = rec_lo + torch.rand_like(rec_lo) * (rec_hi - rec_lo)
                hold = 0.3 + float(torch.rand(1)) * 0.7
            tgt = goal
            with shared.lock:
                for i, n in enumerate(ARM_JOINT_NAMES):
                    shared.target[n] = float(goal[i])
            rate = rspeed
        else:
            rate = speed
        if clamp:
            tgt = torch.maximum(torch.minimum(tgt, rec_hi), rec_lo)
        tgt = torch.maximum(torch.minimum(tgt, soft[:, 1]), soft[:, 0])
        cur = cur + (tgt - cur).clamp(-rate * dt, rate * dt)
        robot.set_joint_position_target(cur.unsqueeze(0), joint_ids=arm_ids)

        action = policy(obs)
        obs, _, dones, extras = env.step(action)
        died = bool(dones.any()) and not bool(extras.get("time_outs", torch.zeros_like(dones)).bool().any())
        if died:
            falls += 1
            cur = robot.data.joint_pos[0, arm_ids].clone()  # the robot respawned: continue from where the arms really are
            start_xy = robot.data.root_pos_w[0, :2].clone()
        contact = sensor.data.net_forces_w[0, foot_ids, :].norm(dim=-1) > CONTACT_N
        air_run = torch.where(~contact, air_run + 1, torch.zeros_like(air_run))
        lifts += int((air_run == max(int(MIN_AIR_S / dt), 1)).sum())

        step += 1
        steps_done += 1
        if time.time() - t_wall > 1.0:
            fps, steps_done, t_wall = steps_done / (time.time() - t_wall), 0, time.time()
        if step % args.stream_every == 0:
            frame = base.render()
            if frame is not None:
                data = encode_jpeg(frame)
                with shared.frame_cond:
                    shared.jpeg, shared.frame_id = data, shared.frame_id + 1
                    shared.frame_cond.notify_all()
        g = robot.data.projected_gravity_b[0, :2].norm().clamp(max=1.0)
        with shared.lock:
            shared.current = {n: float(cur[i]) for i, n in enumerate(ARM_JOINT_NAMES)}
            shared.telemetry = {
                "z": float(robot.data.root_pos_w[0, 2]),
                "tilt": float(torch.rad2deg(torch.asin(g))),
                "drift_cm": float((robot.data.root_pos_w[0, :2] - start_xy).norm() * 100),
                "speed": float(robot.data.root_lin_vel_w[0, :2].norm()),
                "lifts": lifts, "falls": falls, "fps": fps,
            }
        next_t += dt  # real-time pacing at the control rate when the machine is fast enough
        delay = next_t - time.time()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.time()
    server.shutdown()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
