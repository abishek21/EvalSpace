"""Allocentric Reference Frame - 5 Multi-Agent Scenarios"""
import io, os, base64, numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perspective_proto")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def figure_right(yaw_deg):
    yaw = np.radians(yaw_deg)
    return np.array([np.cos(yaw), np.sin(yaw)])

def find_objects_on_side(objects, fig_pos, yaw_deg, side="left"):
    results = []
    right = figure_right(yaw_deg)
    for obj in objects:
        delta = np.array(obj["pos"][:2]) - np.array(fig_pos[:2])
        proj = np.dot(delta, right[:2])
        if side == "right" and proj > 0.08:
            results.append((obj, proj))
        elif side == "left" and proj < -0.08:
            results.append((obj, -proj))
    results.sort(key=lambda x: -x[1])
    return [r[0] for r in results]

def object_in_front(objects, fig_pos, yaw_deg):
    yaw = np.radians(yaw_deg)
    fwd = np.array([-np.sin(yaw), np.cos(yaw)])
    best = None
    best_dist = 999
    for obj in objects:
        delta = np.array(obj["pos"][:2]) - np.array(fig_pos[:2])
        fp = np.dot(delta, fwd)
        dist = np.linalg.norm(delta)
        if fp > 0.08 and dist < best_dist:
            best = obj
            best_dist = dist
    return best

def _figure_xml(name, pos, yaw_deg, body_color, hat_color, scale=1.0):
    x, y = pos
    s = scale
    return f"""
    <body name="{name}" pos="{x} {y} 0" euler="0 0 {yaw_deg}">
      <geom type="box" size="{0.04*s} {0.03*s} {0.015*s}" pos="{-0.02*s} 0 {0.015*s}" rgba="{body_color}"/>
      <geom type="box" size="{0.04*s} {0.03*s} {0.015*s}" pos="{0.02*s} 0 {0.015*s}" rgba="{body_color}"/>
      <geom type="capsule" size="{0.018*s}" fromto="{-0.02*s} 0 {0.03*s}  {-0.015*s} 0 {0.14*s}" rgba="{body_color}"/>
      <geom type="capsule" size="{0.018*s}" fromto="{0.02*s} 0 {0.03*s}   {0.015*s} 0 {0.14*s}" rgba="{body_color}"/>
      <geom type="box" size="{0.035*s} {0.022*s} {0.05*s}" pos="0 0 {0.195*s}" rgba="{body_color}"/>
      <geom type="capsule" size="{0.014*s}" fromto="{-0.035*s} 0 {0.23*s}  {-0.08*s} {0.02*s} {0.15*s}" rgba="{body_color}"/>
      <geom type="capsule" size="{0.014*s}" fromto="{0.035*s} 0 {0.23*s}   {0.08*s} {0.02*s} {0.15*s}" rgba="{body_color}"/>
      <geom type="cylinder" size="{0.012*s} {0.01*s}" pos="0 0 {0.255*s}" rgba="0.82 0.63 0.52 1"/>
      <geom type="sphere" size="{0.032*s}" pos="0 0 {0.295*s}" rgba="0.82 0.63 0.52 1"/>
      <geom type="sphere" size="{0.008*s}" pos="0 {0.03*s} {0.288*s}" rgba="0.88 0.55 0.45 1"/>
      <geom type="sphere" size="{0.006*s}" pos="{-0.012*s} {0.028*s} {0.3*s}" rgba="0.1 0.1 0.1 1"/>
      <geom type="sphere" size="{0.006*s}" pos="{0.012*s} {0.028*s} {0.3*s}" rgba="0.1 0.1 0.1 1"/>
      <geom type="sphere" size="{0.028*s}" pos="0 {-0.005*s} {0.32*s}" rgba="{hat_color}"/>
    </body>"""

OBJ_BUILDERS = {
    "red cone": lambda n, p: f'<body name="{n}" pos="{p[0]} {p[1]} 0"><geom type="cylinder" size="0.06 0.12" pos="0 0 0.12" rgba="0.9 0.15 0.1 1"/><geom type="cylinder" size="0.04 0.04" pos="0 0 0.28" rgba="0.85 0.1 0.08 1"/><geom type="sphere" size="0.03" pos="0 0 0.34" rgba="0.8 0.08 0.05 1"/></body>',
    "blue barrel": lambda n, p: f'<body name="{n}" pos="{p[0]} {p[1]} 0"><geom type="cylinder" size="0.08 0.12" pos="0 0 0.12" rgba="0.12 0.25 0.85 1"/><geom type="cylinder" size="0.082 0.008" pos="0 0 0.04" rgba="0.3 0.3 0.35 1"/><geom type="cylinder" size="0.082 0.008" pos="0 0 0.20" rgba="0.3 0.3 0.35 1"/></body>',
    "green box": lambda n, p: f'<body name="{n}" pos="{p[0]} {p[1]} 0"><geom type="box" size="0.08 0.08 0.08" pos="0 0 0.08" rgba="0.1 0.7 0.15 1"/></body>',
    "yellow ball": lambda n, p: f'<body name="{n}" pos="{p[0]} {p[1]} 0"><geom type="sphere" size="0.07" pos="0 0 0.07" rgba="0.92 0.82 0.1 1"/></body>',
    "white pillar": lambda n, p: f'<body name="{n}" pos="{p[0]} {p[1]} 0"><geom type="cylinder" size="0.05 0.18" pos="0 0 0.18" rgba="0.95 0.95 0.92 1"/><geom type="cylinder" size="0.065 0.015" pos="0 0 0.37" rgba="0.9 0.9 0.87 1"/></body>',
    "orange pyramid": lambda n, p: f'<body name="{n}" pos="{p[0]} {p[1]} 0"><geom type="box" size="0.09 0.09 0.04" pos="0 0 0.04" rgba="1.0 0.55 0.1 1"/><geom type="box" size="0.06 0.06 0.04" pos="0 0 0.12" rgba="0.95 0.50 0.08 1"/><geom type="box" size="0.03 0.03 0.04" pos="0 0 0.20" rgba="0.9 0.45 0.06 1"/></body>',
    "purple cylinder": lambda n, p: f'<body name="{n}" pos="{p[0]} {p[1]} 0"><geom type="cylinder" size="0.06 0.10" pos="0 0 0.10" rgba="0.6 0.15 0.75 1"/></body>',
}

AGENT_A = {"body_color": "0.2 0.5 0.9 1", "hat_color": "0.1 0.35 0.75 1"}
AGENT_B = {"body_color": "0.85 0.1 0.1 1", "hat_color": "0.7 0.05 0.05 1"}
AGENT_C = {"body_color": "0.15 0.7 0.3 1", "hat_color": "0.1 0.55 0.2 1"}

def build_scene(objects, figures):
    obj_xml = "\n".join(OBJ_BUILDERS[o["type"]](o["name"].replace(" ","_"), o["pos"]) for o in objects)
    fig_xml = "\n".join(_figure_xml(f["name"], f["pos"], f["yaw"], f["body_color"], f["hat_color"], f.get("scale",1.0)) for f in figures)
    return f"""<mujoco model="allocentric"><option gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset><texture name="f" type="2d" builtin="checker" rgb1="0.88 0.88 0.85" rgb2="0.75 0.75 0.72" width="512" height="512"/>
    <material name="fm" texture="f" texrepeat="8 8" reflectance="0.05"/></asset>
  <worldbody>
    <light pos="0 -3 5" dir="0 0.4 -0.7" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
    <light pos="3 0 4" dir="-0.4 0 -0.7" diffuse="0.4 0.4 0.42"/>
    <light pos="-3 2 4" dir="0.3 -0.2 -0.6" diffuse="0.35 0.35 0.38"/>
    <geom type="plane" size="5 5 0.01" material="fm"/>
    {obj_xml}
    {fig_xml}
  </worldbody></mujoco>"""

def render_scene(xml, label=""):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=960)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 90
    cam.elevation = -30
    cam.distance = 3.0
    cam.lookat[:] = [0, 0, 0.15]
    renderer.update_scene(data, cam)
    pixels = renderer.render()
    renderer.close()
    img = Image.fromarray(pixels)
    if label:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((8,8), label, font=font)
        draw.rectangle([bbox[0]-3, bbox[1]-3, bbox[2]+3, bbox[3]+3], fill=(0,0,0))
        draw.text((8,8), label, fill=(255,255,255), font=font)
    return img

if __name__ == "__main__":
    print("=" * 55, flush=True)
    print("ALLOCENTRIC - 5 Multi-Agent Scenarios", flush=True)
    print("=" * 55, flush=True)

    # S1: 3 agents in triangle facing center
    objs1 = [
        {"name":"red cone","type":"red cone","pos":[-0.5,0.5]},
        {"name":"blue barrel","type":"blue barrel","pos":[0.5,0.5]},
        {"name":"green box","type":"green box","pos":[0.5,-0.4]},
        {"name":"yellow ball","type":"yellow ball","pos":[-0.5,-0.4]},
    ]
    figs1 = [
        {"name":"agent_blue","pos":[-0.6,0.0],"yaw":270,**AGENT_A},
        {"name":"agent_red","pos":[0.6,0.0],"yaw":90,**AGENT_B},
        {"name":"agent_green","pos":[0.0,0.6],"yaw":180,**AGENT_C},
    ]
    left1 = find_objects_on_side(objs1, figs1[1]["pos"], figs1[1]["yaw"], "left")
    q1 = "There are 3 agents and 4 objects. Which object is to the red agent's left? Answer with just the object name."
    a1 = left1[0]["name"] if left1 else "none"
    print(f"\nS1: Q={q1}", flush=True)
    print(f"    GT={a1}  ({[o['name'] for o in left1]})", flush=True)
    xml1 = build_scene(objs1, figs1)
    render_scene(xml1, "allo_m01").save(os.path.join(OUTPUT_DIR, "allo_m01.png"))
    print("    Saved allo_m01.png", flush=True)

    # S2: 2 agents facing each other
    objs2 = [
        {"name":"white pillar","type":"white pillar","pos":[0.0,0.7]},
        {"name":"orange pyramid","type":"orange pyramid","pos":[0.0,-0.7]},
        {"name":"blue barrel","type":"blue barrel","pos":[-0.7,0.0]},
        {"name":"red cone","type":"red cone","pos":[0.7,0.0]},
    ]
    figs2 = [
        {"name":"agent_blue","pos":[-0.25,0.0],"yaw":270,**AGENT_A},
        {"name":"agent_red","pos":[0.25,0.0],"yaw":90,**AGENT_B},
    ]
    left2 = find_objects_on_side(objs2, figs2[0]["pos"], figs2[0]["yaw"], "left")
    q2 = "Two agents face each other. Which object is to the blue agent's left? Answer with just the object name."
    a2 = left2[0]["name"] if left2 else "none"
    print(f"\nS2: Q={q2}", flush=True)
    print(f"    GT={a2}  ({[o['name'] for o in left2]})", flush=True)
    xml2 = build_scene(objs2, figs2)
    render_scene(xml2, "allo_m02").save(os.path.join(OUTPUT_DIR, "allo_m02.png"))
    print("    Saved allo_m02.png", flush=True)

    # S3: 3 agents in line facing same direction, ask about middle
    objs3 = [
        {"name":"yellow ball","type":"yellow ball","pos":[-0.6,0.5]},
        {"name":"green box","type":"green box","pos":[0.0,0.6]},
        {"name":"purple cylinder","type":"purple cylinder","pos":[0.6,0.5]},
        {"name":"red cone","type":"red cone","pos":[0.0,-0.5]},
    ]
    figs3 = [
        {"name":"agent_blue","pos":[-0.4,0.0],"yaw":0,**AGENT_A},
        {"name":"agent_red","pos":[0.0,0.0],"yaw":0,**AGENT_B},
        {"name":"agent_green","pos":[0.4,0.0],"yaw":0,**AGENT_C},
    ]
    front3 = object_in_front(objs3, figs3[1]["pos"], figs3[1]["yaw"])
    q3 = "Three agents stand in a line all facing the same direction. Which object is directly in front of the red agent? Answer with just the object name."
    a3 = front3["name"] if front3 else "none"
    print(f"\nS3: Q={q3}", flush=True)
    print(f"    GT={a3}", flush=True)
    xml3 = build_scene(objs3, figs3)
    render_scene(xml3, "allo_m03").save(os.path.join(OUTPUT_DIR, "allo_m03.png"))
    print("    Saved allo_m03.png", flush=True)

    # S4: 2 agents back to back, blue faces viewer
    objs4 = [
        {"name":"blue barrel","type":"blue barrel","pos":[-0.5,0.5]},
        {"name":"orange pyramid","type":"orange pyramid","pos":[0.5,0.5]},
        {"name":"white pillar","type":"white pillar","pos":[0.5,-0.5]},
        {"name":"yellow ball","type":"yellow ball","pos":[-0.5,-0.5]},
    ]
    figs4 = [
        {"name":"agent_blue","pos":[0.0,0.15],"yaw":180,**AGENT_A},
        {"name":"agent_red","pos":[0.0,-0.15],"yaw":0,**AGENT_B},
    ]
    right4 = find_objects_on_side(objs4, figs4[0]["pos"], figs4[0]["yaw"], "right")
    q4 = "Two agents stand back to back. Which object is to the blue agent's right? Answer with just the object name."
    a4 = right4[0]["name"] if right4 else "none"
    print(f"\nS4: Q={q4}", flush=True)
    print(f"    GT={a4}  (blue faces -Y, right=-X=viewer LEFT!) ({[o['name'] for o in right4]})", flush=True)
    xml4 = build_scene(objs4, figs4)
    render_scene(xml4, "allo_m04").save(os.path.join(OUTPUT_DIR, "allo_m04.png"))
    print("    Saved allo_m04.png", flush=True)

    # S5: 3 agents in circle facing outward (120 degree angles)
    objs5 = [
        {"name":"red cone","type":"red cone","pos":[0.0,0.65]},
        {"name":"green box","type":"green box","pos":[0.56,-0.32]},
        {"name":"white pillar","type":"white pillar","pos":[-0.56,-0.32]},
        {"name":"yellow ball","type":"yellow ball","pos":[0.0,-0.6]},
    ]
    figs5 = [
        {"name":"agent_blue","pos":[0.0,0.35],"yaw":0,**AGENT_A},
        {"name":"agent_red","pos":[0.3,-0.18],"yaw":240,**AGENT_B},
        {"name":"agent_green","pos":[-0.3,-0.18],"yaw":120,**AGENT_C},
    ]
    right5 = find_objects_on_side(objs5, figs5[2]["pos"], figs5[2]["yaw"], "right")
    q5 = "Three agents stand in a circle facing outward. Which object is to the green agent's right? Answer with just the object name."
    a5 = right5[0]["name"] if right5 else "none"
    print(f"\nS5: Q={q5}", flush=True)
    print(f"    GT={a5}  ({[o['name'] for o in right5]})", flush=True)
    xml5 = build_scene(objs5, figs5)
    render_scene(xml5, "allo_m05").save(os.path.join(OUTPUT_DIR, "allo_m05.png"))
    print("    Saved allo_m05.png", flush=True)

    print(f"\nAll images in: {OUTPUT_DIR}/", flush=True)


def publish():
    """Publish the 5 allocentric scenarios to the database."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import db
    from uuid import uuid4
    from datetime import datetime
    
    dataset_id = str(uuid4())
    
    # Re-generate all scenarios with images
    scenarios_data = [
        {"id": "allo_m01", "objs": [
            {"name":"red cone","type":"red cone","pos":[-0.5,0.5]},
            {"name":"blue barrel","type":"blue barrel","pos":[0.5,0.5]},
            {"name":"green box","type":"green box","pos":[0.5,-0.4]},
            {"name":"yellow ball","type":"yellow ball","pos":[-0.5,-0.4]},
        ], "figs": [
            {"name":"agent_blue","pos":[-0.6,0.0],"yaw":270,**AGENT_A},
            {"name":"agent_red","pos":[0.6,0.0],"yaw":90,**AGENT_B},
            {"name":"agent_green","pos":[0.0,0.6],"yaw":180,**AGENT_C},
        ], "q": "There are 3 agents and 4 objects. Which object is to the red agent's left? Answer with just the object name.",
           "side": "left", "fig_idx": 1},
        
        {"id": "allo_m02", "objs": [
            {"name":"white pillar","type":"white pillar","pos":[0.0,0.7]},
            {"name":"orange pyramid","type":"orange pyramid","pos":[0.0,-0.7]},
            {"name":"blue barrel","type":"blue barrel","pos":[-0.7,0.0]},
            {"name":"red cone","type":"red cone","pos":[0.7,0.0]},
        ], "figs": [
            {"name":"agent_blue","pos":[-0.25,0.0],"yaw":270,**AGENT_A},
            {"name":"agent_red","pos":[0.25,0.0],"yaw":90,**AGENT_B},
        ], "q": "Two agents face each other. Which object is to the blue agent's left? Answer with just the object name.",
           "side": "left", "fig_idx": 0},
        
        {"id": "allo_m03", "objs": [
            {"name":"yellow ball","type":"yellow ball","pos":[-0.6,0.5]},
            {"name":"green box","type":"green box","pos":[0.0,0.6]},
            {"name":"purple cylinder","type":"purple cylinder","pos":[0.6,0.5]},
            {"name":"red cone","type":"red cone","pos":[0.0,-0.5]},
        ], "figs": [
            {"name":"agent_blue","pos":[-0.4,0.0],"yaw":0,**AGENT_A},
            {"name":"agent_red","pos":[0.0,0.0],"yaw":0,**AGENT_B},
            {"name":"agent_green","pos":[0.4,0.0],"yaw":0,**AGENT_C},
        ], "q": "Three agents stand in a line all facing the same direction. Which object is directly in front of the red agent? Answer with just the object name.",
           "side": "front", "fig_idx": 1},
        
        {"id": "allo_m04", "objs": [
            {"name":"blue barrel","type":"blue barrel","pos":[-0.5,0.5]},
            {"name":"orange pyramid","type":"orange pyramid","pos":[0.5,0.5]},
            {"name":"white pillar","type":"white pillar","pos":[0.5,-0.5]},
            {"name":"yellow ball","type":"yellow ball","pos":[-0.5,-0.5]},
        ], "figs": [
            {"name":"agent_blue","pos":[0.0,0.15],"yaw":180,**AGENT_A},
            {"name":"agent_red","pos":[0.0,-0.15],"yaw":0,**AGENT_B},
        ], "q": "Two agents stand back to back. Which object is to the blue agent's right? Answer with just the object name.",
           "side": "right", "fig_idx": 0},
        
        {"id": "allo_m05", "objs": [
            {"name":"red cone","type":"red cone","pos":[0.0,0.65]},
            {"name":"green box","type":"green box","pos":[0.56,-0.32]},
            {"name":"white pillar","type":"white pillar","pos":[-0.56,-0.32]},
            {"name":"yellow ball","type":"yellow ball","pos":[0.0,-0.6]},
        ], "figs": [
            {"name":"agent_blue","pos":[0.0,0.35],"yaw":0,**AGENT_A},
            {"name":"agent_red","pos":[0.3,-0.18],"yaw":240,**AGENT_B},
            {"name":"agent_green","pos":[-0.3,-0.18],"yaw":120,**AGENT_C},
        ], "q": "Three agents stand in a circle facing outward. Which object is to the green agent's right? Answer with just the object name.",
           "side": "right", "fig_idx": 2},
    ]
    
    all_records = []
    print("Publishing allocentric-5 scenarios...", flush=True)
    
    for sc in scenarios_data:
        fig = sc["figs"][sc["fig_idx"]]
        if sc["side"] == "front":
            answer_obj = object_in_front(sc["objs"], fig["pos"], fig["yaw"])
        else:
            objs_on_side = find_objects_on_side(sc["objs"], fig["pos"], fig["yaw"], sc["side"])
            answer_obj = objs_on_side[0] if objs_on_side else None
        
        answer = answer_obj["name"] if answer_obj else "none"
        
        # Render
        xml = build_scene(sc["objs"], sc["figs"])
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=720, width=960)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = 90
        cam.elevation = -30
        cam.distance = 3.0
        cam.lookat[:] = [0, 0, 0.15]
        renderer.update_scene(data, cam)
        pixels = renderer.render()
        renderer.close()
        
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="JPEG", quality=90)
        scene_image = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        
        record = {
            "id": str(uuid4()),
            "dataset_id": dataset_id,
            "pair_type": "ground_truth",
            "scene_id": sc["id"],
            "prompt": sc["q"],
            "category": "allocentric",
            "difficulty": "hard",
            "ground_truth": {
                "answer": answer,
                "question_type": sc["side"],
                "agent_queried": fig["name"],
                "agent_yaw": fig["yaw"],
            },
            "source": {
                "dataset": "mujoco:allocentric",
                "scene_id": sc["id"],
                "images": [scene_image],
            },
            "status": "ready",
        }
        all_records.append(record)
        print(f"  {sc['id']}: {sc['q'][:60]}... -> {answer}", flush=True)
    
    db.create_dataset({
        "id": dataset_id,
        "name": "allocentric-5",
        "task_type": "allocentric",
        "scenario_count": 5,
        "created_at": datetime.now().isoformat(),
        "config": {
            "environment": "allocentric",
            "mode": "curated",
            "answer_format": "free_form",
        },
    })
    db.add_scenarios(all_records)
    
    print(f"\n{'='*55}", flush=True)
    print(f"Published: allocentric-5", flush=True)
    print(f"Dataset ID: {dataset_id}", flush=True)
    print(f"Scenarios: 5", flush=True)
    print(f"{'='*55}", flush=True)
    return dataset_id

if "--publish" in __import__("sys").argv:
    publish()
