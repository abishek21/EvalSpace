"""Quick demo: render MuJoCo simulation frames into an HTML viewer."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.mujoco_sim import create_quick_scene, simulate_trajectory, Waypoint, render_scene
import mujoco

# Build scene: cup, box, bottle on a table
xml, _ = create_quick_scene([
    {"name": "cup1", "type": "cup", "pos": [0.0, 0.0, 0.42]},
    {"name": "box1", "type": "box", "pos": [0.2, 0.0, 0.42]},
    {"name": "bottle1", "type": "bottle", "pos": [-0.15, 0.1, 0.47]},
    {"name": "ball1", "type": "ball", "pos": [0.1, -0.15, 0.42]},
])

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Let objects settle
for _ in range(500):
    mujoco.mj_step(model, data)

# Render initial state from 3 views
print("Rendering initial views...")
views_before = {
    cam: render_scene(model, data, cam, 640, 480)
    for cam in ["front", "top", "side"]
}

# Simulate: move cup toward box
print("Simulating trajectory...")
result = simulate_trajectory(
    xml, "cup1",
    waypoints=[
        Waypoint(0.0, 0.0, 0.42, 0.0),
        Waypoint(0.1, 0.0, 0.42, 0.5),
        Waypoint(0.2, 0.0, 0.42, 1.0),
        Waypoint(0.2, 0.0, 0.42, 1.5),
    ],
    record_interval=0.1,
    record_cameras=["front"],
)

# Render final state
model2 = mujoco.MjModel.from_xml_string(xml)
data2 = mujoco.MjData(model2)
# Set cup to final position and step
for _ in range(500):
    mujoco.mj_step(model2, data2)
jnt_id = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_JOINT, "cup1_jnt")
qpos_adr = model2.jnt_qposadr[jnt_id]
data2.qpos[qpos_adr:qpos_adr+3] = [0.2, 0.0, 0.42]
for _ in range(300):
    mujoco.mj_step(model2, data2)

views_after = {
    cam: render_scene(model2, data2, cam, 640, 480)
    for cam in ["front", "top", "side"]
}

# Build HTML
frames_js = ",\n".join(f'"{f}"' for f in result.frames)
collisions_html = ""
for c in result.collisions[:5]:
    collisions_html += f'<tr><td>{c["time"]}s</td><td>{c["object"]}</td><td>{c["other"]}</td><td>{c["force"]}</td></tr>\n'

html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>MuJoCo Simulation Demo</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; color: #fff; }}
  h2 {{ font-size: 20px; margin: 24px 0 12px; color: #a0a0ff; }}
  h3 {{ font-size: 16px; margin: 16px 0 8px; color: #80d0ff; }}
  .subtitle {{ color: #888; margin-bottom: 24px; }}
  .views {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .views img {{ border-radius: 8px; border: 1px solid #333; max-width: 320px; }}
  .views .label {{ text-align: center; color: #888; font-size: 12px; margin-top: 4px; }}
  .view-card {{ text-align: center; }}
  .player {{ text-align: center; margin: 16px 0; }}
  .player img {{ border-radius: 8px; border: 2px solid #444; max-width: 480px; }}
  .controls {{ margin: 12px 0; }}
  .controls button {{ background: #2a2a3a; color: #fff; border: 1px solid #555; padding: 8px 20px; border-radius: 6px; cursor: pointer; margin: 0 4px; font-size: 14px; }}
  .controls button:hover {{ background: #3a3a5a; }}
  .slider {{ width: 400px; margin: 8px; }}
  table {{ border-collapse: collapse; margin: 8px 0; }}
  th, td {{ padding: 6px 16px; border: 1px solid #333; text-align: left; font-size: 14px; }}
  th {{ background: #1a1a2a; color: #a0a0ff; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 600; }}
  .badge.collision {{ background: #ff4444; color: #fff; }}
  .badge.clear {{ background: #44cc44; color: #000; }}
  .stats {{ display: flex; gap: 24px; margin: 16px 0; }}
  .stat {{ background: #1a1a2a; padding: 16px 24px; border-radius: 8px; text-align: center; }}
  .stat .val {{ font-size: 28px; font-weight: bold; color: #fff; }}
  .stat .lbl {{ font-size: 12px; color: #888; margin-top: 4px; }}
  .section {{ background: #111; border-radius: 12px; padding: 20px; margin: 16px 0; border: 1px solid #222; }}
</style>
</head><body>

<h1>🤖 MuJoCo Physics Verification</h1>
<p class="subtitle">Trajectory simulation for spatial reasoning RLHF</p>

<div class="stats">
  <div class="stat"><div class="val">{len(result.collisions)}</div><div class="lbl">Collisions</div></div>
  <div class="stat"><div class="val">{len(result.trajectory_actual)}</div><div class="lbl">Trajectory Points</div></div>
  <div class="stat"><div class="val">{len(result.frames)}</div><div class="lbl">Frames</div></div>
  <div class="stat"><div class="val">{'✅' if result.physics_plausible else '❌'}</div><div class="lbl">Physics Valid</div></div>
</div>

<div class="section">
  <h2>📷 Initial Scene (3 Views)</h2>
  <div class="views">
    <div class="view-card"><img src="{views_before['front']}"/><div class="label">Front</div></div>
    <div class="view-card"><img src="{views_before['top']}"/><div class="label">Top-down</div></div>
    <div class="view-card"><img src="{views_before['side']}"/><div class="label">Side</div></div>
  </div>
</div>

<div class="section">
  <h2>🎬 Trajectory Playback</h2>
  <p>Moving <b>cup1</b> → toward <b>box1</b> (0.0 → 0.2 on X axis)</p>
  <div class="player">
    <img id="frame" src="{result.frames[0] if result.frames else ''}" />
    <div class="controls">
      <button onclick="play()">▶ Play</button>
      <button onclick="pause()">⏸ Pause</button>
      <button onclick="reset()">⏮ Reset</button>
    </div>
    <input type="range" class="slider" id="slider" min="0" max="{len(result.frames)-1}" value="0" oninput="showFrame(this.value)"/>
    <div id="timeLabel" style="color:#888; font-size:13px;">Frame 0 / {len(result.frames)-1}</div>
  </div>
</div>

<div class="section">
  <h2>💥 Collision Report</h2>
  <span class="badge collision">⚠️ {len(result.collisions)} collisions detected</span>
  <p style="margin:8px 0; color:#aaa;">{result.reason}</p>
  <table>
    <tr><th>Time</th><th>Object</th><th>Hit</th><th>Force</th></tr>
    {collisions_html}
  </table>
</div>

<div class="section">
  <h2>📷 Final Scene (After Simulation)</h2>
  <div class="views">
    <div class="view-card"><img src="{views_after['front']}"/><div class="label">Front</div></div>
    <div class="view-card"><img src="{views_after['top']}"/><div class="label">Top-down</div></div>
    <div class="view-card"><img src="{views_after['side']}"/><div class="label">Side</div></div>
  </div>
</div>

<script>
const frames = [{frames_js}];
let idx = 0, timer = null;
const img = document.getElementById('frame');
const slider = document.getElementById('slider');
const label = document.getElementById('timeLabel');

function showFrame(i) {{
  idx = parseInt(i);
  if (frames[idx]) img.src = frames[idx];
  slider.value = idx;
  label.textContent = `Frame ${{idx}} / ${{frames.length-1}}`;
}}

function play() {{
  pause();
  timer = setInterval(() => {{
    if (idx >= frames.length - 1) {{ pause(); return; }}
    showFrame(idx + 1);
  }}, 150);
}}

function pause() {{ clearInterval(timer); timer = null; }}
function reset() {{ pause(); showFrame(0); }}
</script>
</body></html>"""

out_path = os.path.join(os.path.dirname(__file__), "mujoco_demo.html")
with open(out_path, "w") as f:
    f.write(html)
print(f"Saved to {out_path}")
