"""Quick render preview to check quality improvements."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.dise_collision import generate_10_scenarios, simulate_collision

print("Generating collision scenarios with upgraded rendering...")
scenarios = generate_10_scenarios()

html = """<!DOCTYPE html><html><head><style>
body { font-family: Inter, sans-serif; background: #f8fafc; margin: 40px; }
h1 { color: #0f172a; }
h2 { color: #334155; font-size: 18px; margin: 32px 0 8px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 1200px; margin-bottom: 16px; }
.card { background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.card img { width: 100%; display: block; }
.card p { padding: 12px; font-size: 13px; color: #64748b; }
.result { font-weight: 600; }
.hit { color: #dc2626; }
.miss { color: #059669; }
</style></head><body>
<h1>🎱 Rendering Quality Preview</h1>
<p style="color:#64748b;">1024×768, 4096 shadows, 3-point lighting, shiny materials, 8× multisampling, spaced objects</p>
"""

# Render scenarios 1, 6 (with blocker), 9 (narrow gap)
for i in [0, 5, 8]:
    s = scenarios[i]
    result = simulate_collision(s)
    label = "HIT" if result.hit_target else "MISS"
    cls = "hit" if result.hit_target else "miss"
    html += f'<h2>{s.name} — <span class="result {cls}">{label}</span></h2>\n'
    html += f'<p style="color:#475569; font-size:14px;">{s.question}</p>\n'
    html += '<div class="grid">\n'
    labels = ["Front", "Angle", "Top", "Side"]
    for j, img in enumerate(result.before_images):
        html += f'<div class="card"><img src="{img}"/><p>{labels[j]}</p></div>\n'
    html += '</div>\n'

html += "</body></html>"

with open("quality_preview.html", "w") as f:
    f.write(html)

print(f"✅ Saved quality_preview.html with 3 scenarios")
