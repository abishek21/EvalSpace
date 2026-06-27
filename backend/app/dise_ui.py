"""
DISE UI: Visual Coding Evaluation Environment

Generates random UI scenes (HTML/CSS), renders screenshots via Playwright,
evaluates VLM ability to modify code based on visual instructions.

Pipeline:
  1. Generate world JSON → render to HTML/CSS → Playwright screenshot
  2. Create instruction ("Move the logo to the top-right corner")
  3. Send screenshot + original code + instruction to VLM
  4. VLM outputs modified HTML/CSS
  5. Render VLM's code → screenshot + constraint checks → score

Constraint types (generic, composable):
  - position: element bounding box in target region
  - center:   element centered on axis
  - spacing:  gap between elements > threshold
  - size:     element dimensions match requirement
  - visible:  element exists and is visible
"""

import asyncio
import base64
import io
import json
import random
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from PIL import Image


# ─── World Elements ──────────────────────────────────────────────────

COLORS = {
    "blue":    "#3b82f6",
    "red":     "#ef4444",
    "green":   "#22c55e",
    "purple":  "#8b5cf6",
    "orange":  "#f97316",
    "pink":    "#ec4899",
    "gray":    "#6b7280",
    "slate":   "#475569",
    "teal":    "#14b8a6",
    "indigo":  "#6366f1",
}

LOGO_SHAPES = [
    {"type": "circle", "label": "circle logo"},
    {"type": "rounded-rect", "label": "rounded logo"},
    {"type": "square", "label": "square logo"},
]

BUTTON_LABELS = [
    "Get Started", "Sign Up", "Learn More", "Subscribe",
    "Download", "Contact Us", "Buy Now", "Try Free",
]

TITLE_TEXTS = [
    "Welcome to Our Platform",
    "Build Something Amazing",
    "The Future of Design",
    "Start Your Journey",
    "Powerful & Simple",
    "Create Without Limits",
]

CARD_TITLES = [
    "Analytics", "Security", "Performance",
    "Integration", "Support", "Automation",
]

ICON_TYPES = ["star", "heart", "bolt", "shield", "chart", "globe"]

VIEWPORT = {"width": 1280, "height": 720}


# ─── Data Classes ────────────────────────────────────────────────────

@dataclass
class UIElement:
    """A single UI element in the scene."""
    id: str
    el_type: str          # logo, button, title, card, icon, nav, text
    x: int
    y: int
    width: int
    height: int
    content: str = ""
    color: str = "#3b82f6"
    bg_color: str = ""
    font_size: int = 16
    extra: dict = field(default_factory=dict)


@dataclass
class Constraint:
    """A single evaluation constraint."""
    type: str             # position, center, spacing, size, visible
    selector: str         # CSS selector (#logo, .card, etc.)
    params: dict = field(default_factory=dict)
    description: str = ""


@dataclass
class UIScenario:
    """Complete UI evaluation scenario."""
    scene_id: str
    elements: list        # list of UIElement
    instruction: str      # what the VLM should do
    constraints: list     # list of Constraint
    difficulty: str = "medium"
    task_type: str = ""   # position, center, spacing, etc.


# ─── HTML/CSS Renderer ──────────────────────────────────────────────

def _element_to_html(el: UIElement) -> str:
    """Convert a UIElement to an HTML string with inline styles."""
    base = (
        f"position:absolute; left:{el.x}px; top:{el.y}px; "
        f"width:{el.width}px; height:{el.height}px; "
        f"box-sizing:border-box; "
    )

    if el.el_type == "logo":
        shape = el.extra.get("shape", "circle")
        radius = "50%" if shape == "circle" else "12px" if shape == "rounded-rect" else "4px"
        return (
            f'<div id="{el.id}" class="logo" style="{base}'
            f'background:{el.color}; border-radius:{radius}; '
            f'display:flex; align-items:center; justify-content:center; '
            f'color:white; font-weight:700; font-size:{el.font_size}px;">'
            f'{el.content}</div>'
        )

    if el.el_type == "button":
        return (
            f'<button id="{el.id}" class="btn" style="{base}'
            f'background:{el.color}; color:white; border:none; '
            f'border-radius:8px; font-size:{el.font_size}px; '
            f'font-weight:600; cursor:pointer; '
            f'display:flex; align-items:center; justify-content:center;">'
            f'{el.content}</button>'
        )

    if el.el_type == "title":
        return (
            f'<h1 id="{el.id}" class="title" style="{base}'
            f'font-size:{el.font_size}px; font-weight:800; '
            f'color:{el.color}; margin:0; display:flex; align-items:center;">'
            f'{el.content}</h1>'
        )

    if el.el_type == "card":
        icon_svg = _icon_svg(el.extra.get("icon", "star"), el.color)
        return (
            f'<div id="{el.id}" class="card" style="{base}'
            f'background:white; border:1px solid #e5e7eb; border-radius:12px; '
            f'padding:24px; display:flex; flex-direction:column; gap:12px;">'
            f'<div style="width:40px;height:40px;">{icon_svg}</div>'
            f'<div style="font-size:18px;font-weight:700;color:#1f2937">{el.content}</div>'
            f'<div style="font-size:14px;color:#6b7280;line-height:1.5">'
            f'A brief description of the {el.content.lower()} feature.</div>'
            f'</div>'
        )

    if el.el_type == "icon":
        svg = _icon_svg(el.extra.get("icon_type", "star"), el.color)
        return (
            f'<div id="{el.id}" class="icon" style="{base}'
            f'display:flex; align-items:center; justify-content:center;">'
            f'{svg}</div>'
        )

    if el.el_type == "nav":
        links = el.extra.get("links", ["Home", "About", "Contact"])
        links_html = " ".join(
            f'<a style="color:{el.color};text-decoration:none;font-size:14px;font-weight:500">{l}</a>'
            for l in links
        )
        return (
            f'<nav id="{el.id}" class="nav" style="{base}'
            f'display:flex; align-items:center; gap:24px; '
            f'padding:0 24px;">{links_html}</nav>'
        )

    if el.el_type == "text":
        return (
            f'<p id="{el.id}" class="text" style="{base}'
            f'font-size:{el.font_size}px; color:{el.color}; '
            f'margin:0; line-height:1.6;">{el.content}</p>'
        )

    # Fallback: colored div
    return (
        f'<div id="{el.id}" style="{base}'
        f'background:{el.color}; border-radius:8px;"></div>'
    )


def _icon_svg(icon_type: str, color: str) -> str:
    """Simple SVG icons."""
    icons = {
        "star":   f'<svg width="40" height="40" viewBox="0 0 24 24" fill="{color}"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
        "heart":  f'<svg width="40" height="40" viewBox="0 0 24 24" fill="{color}"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>',
        "bolt":   f'<svg width="40" height="40" viewBox="0 0 24 24" fill="{color}"><path d="M13 2L3 14h9l-1 10 10-12h-9l1-10z"/></svg>',
        "shield": f'<svg width="40" height="40" viewBox="0 0 24 24" fill="{color}"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/></svg>',
        "chart":  f'<svg width="40" height="40" viewBox="0 0 24 24" fill="{color}"><path d="M3 13h2v8H3v-8zm4-6h2v14H7V7zm4-4h2v18h-2V3zm4 8h2v10h-2V11zm4-3h2v13h-2V8z"/></svg>',
        "globe":  f'<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15 15 0 010 20M12 2a15 15 0 000 20"/></svg>',
    }
    return icons.get(icon_type, icons["star"])


def render_html(elements: list[UIElement], bg: str = "#f8fafc") -> str:
    """Render a list of UIElements to a complete HTML page."""
    els_html = "\n    ".join(_element_to_html(el) for el in elements)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    width: {VIEWPORT['width']}px;
    height: {VIEWPORT['height']}px;
    background: {bg};
    font-family: -apple-system, 'Inter', 'Segoe UI', sans-serif;
    position: relative;
    overflow: hidden;
  }}
</style>
</head>
<body>
    {els_html}
</body>
</html>"""


# ─── Playwright Screenshot ───────────────────────────────────────────

async def _screenshot_async(html: str, width: int = 1280, height: int = 720) -> bytes:
    """Render HTML to PNG bytes using Playwright."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.set_content(html, wait_until="networkidle")
        png = await page.screenshot(type="png")
        await browser.close()
    return png


def screenshot(html: str, width: int = 1280, height: int = 720) -> str:
    """Render HTML → base64 PNG string (sync wrapper)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                png = pool.submit(
                    lambda: asyncio.run(_screenshot_async(html, width, height))
                ).result(timeout=30)
        else:
            png = loop.run_until_complete(_screenshot_async(html, width, height))
    except RuntimeError:
        png = asyncio.run(_screenshot_async(html, width, height))

    return "data:image/png;base64," + base64.b64encode(png).decode()


# ─── Constraint Evaluation ───────────────────────────────────────────

async def _evaluate_constraints_async(
    html: str, constraints: list[Constraint],
    width: int = 1280, height: int = 720,
) -> list[dict]:
    """
    Render HTML in Playwright and check each constraint.
    Returns list of {constraint, passed, actual, detail}.
    """
    from playwright.async_api import async_playwright

    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.set_content(html, wait_until="networkidle")

        for c in constraints:
            try:
                result = await _check_one(page, c, width, height)
                results.append(result)
            except Exception as e:
                results.append({
                    "constraint": c.description or c.type,
                    "passed": False,
                    "actual": str(e),
                    "detail": f"Error: {e}",
                })

        await browser.close()
    return results


async def _check_one(page, c: Constraint, vw: int, vh: int) -> dict:
    """Check a single constraint against the rendered page."""

    # Types that don't need a main selector
    if c.type == "equal_spacing":
        selectors = c.params.get("selectors", [])
        tolerance = c.params.get("tolerance", 15)
        boxes = []
        for sel in selectors:
            b = await page.evaluate(f'''(() => {{
                const el = document.querySelector('{sel}');
                if (!el) return null;
                const r = el.getBoundingClientRect();
                return {{x: r.x, y: r.y, right: r.right, bottom: r.bottom}};
            }})()''')
            if b:
                boxes.append(b)
        if len(boxes) < 2:
            return {"constraint": c.description, "passed": False, "actual": f"Only {len(boxes)} elements found", "detail": "Need at least 2"}
        gaps = []
        for i in range(len(boxes) - 1):
            gap = boxes[i+1]["y"] - boxes[i]["bottom"]
            if gap < 0:
                gap = boxes[i+1]["x"] - boxes[i]["right"]
            gaps.append(abs(gap))
        if len(gaps) >= 2:
            spread = max(gaps) - min(gaps)
            passed = spread <= tolerance
            detail = f"gaps={[f'{g:.0f}' for g in gaps]}, spread={spread:.0f}px (tol={tolerance})"
        else:
            passed = True
            detail = f"single gap={gaps[0]:.0f}px"
        return {"constraint": c.description or "equal_spacing", "passed": passed, "actual": gaps, "detail": detail}

    if c.type == "contains_text":
        inner = await page.evaluate(f'''(() => {{
            const el = document.querySelector('{c.selector}');
            return el ? el.innerText || el.textContent || '' : '';
        }})()''')
        text = c.params.get("text", "")
        passed = text.lower() in (inner or "").lower()
        detail = f"looking for '{text}' in '{(inner or '')[:60]}' — {'found' if passed else 'not found'}"
        return {"constraint": c.description or "contains_text", "passed": passed, "actual": inner, "detail": detail}

    # Get bounding box via JS
    bbox = await page.evaluate(f'''(() => {{
        const el = document.querySelector('{c.selector}');
        if (!el) return null;
        const r = el.getBoundingClientRect();
        return {{x: r.x, y: r.y, width: r.width, height: r.height,
                 right: r.right, bottom: r.bottom,
                 cx: r.x + r.width/2, cy: r.y + r.height/2}};
    }})()''')

    if bbox is None:
        return {
            "constraint": c.description or c.type,
            "passed": False,
            "actual": "Element not found",
            "detail": f"Selector '{c.selector}' not found in DOM",
        }

    passed = False
    detail = ""

    if c.type == "position":
        region = c.params.get("region", "")
        margin = c.params.get("margin", 60)

        if region == "top-right":
            passed = bbox["right"] > vw - margin and bbox["y"] < margin
            detail = f"right={bbox['right']:.0f} (need >{vw - margin}), top={bbox['y']:.0f} (need <{margin})"
        elif region == "top-left":
            passed = bbox["x"] < margin and bbox["y"] < margin
            detail = f"left={bbox['x']:.0f} (need <{margin}), top={bbox['y']:.0f} (need <{margin})"
        elif region == "bottom-right":
            passed = bbox["right"] > vw - margin and bbox["bottom"] > vh - margin
            detail = f"right={bbox['right']:.0f}, bottom={bbox['bottom']:.0f}"
        elif region == "bottom-left":
            passed = bbox["x"] < margin and bbox["bottom"] > vh - margin
            detail = f"left={bbox['x']:.0f}, bottom={bbox['bottom']:.0f}"
        elif region == "bottom-center":
            passed = abs(bbox["cx"] - vw / 2) < margin and bbox["bottom"] > vh - margin
            detail = f"cx={bbox['cx']:.0f} (need ~{vw//2}), bottom={bbox['bottom']:.0f}"
        else:
            detail = f"Unknown region: {region}"

    elif c.type == "center":
        axis = c.params.get("axis", "horizontal")
        tolerance = c.params.get("tolerance", 40)
        if axis == "horizontal":
            passed = abs(bbox["cx"] - vw / 2) < tolerance
            detail = f"cx={bbox['cx']:.0f}, viewport_center={vw//2}, diff={abs(bbox['cx'] - vw/2):.0f} (tol={tolerance})"
        elif axis == "vertical":
            passed = abs(bbox["cy"] - vh / 2) < tolerance
            detail = f"cy={bbox['cy']:.0f}, viewport_center={vh//2}, diff={abs(bbox['cy'] - vh/2):.0f}"
        elif axis == "both":
            passed = abs(bbox["cx"] - vw / 2) < tolerance and abs(bbox["cy"] - vh / 2) < tolerance
            detail = f"cx={bbox['cx']:.0f}, cy={bbox['cy']:.0f}, center=({vw//2},{vh//2})"

    elif c.type == "spacing":
        sel2 = c.params.get("selector2", "")
        min_gap = c.params.get("min_gap", 20)
        bbox2 = await page.evaluate(f'''(() => {{
            const el = document.querySelector('{sel2}');
            if (!el) return null;
            const r = el.getBoundingClientRect();
            return {{x: r.x, y: r.y, width: r.width, height: r.height, right: r.right, bottom: r.bottom}};
        }})()''')
        if bbox2 is None:
            return {"constraint": c.description, "passed": False, "actual": f"'{sel2}' not found", "detail": ""}
        # Check horizontal or vertical gap
        h_gap = max(0, max(bbox2["x"] - bbox["right"], bbox["x"] - bbox2["right"]))
        v_gap = max(0, max(bbox2["y"] - bbox["bottom"], bbox["y"] - bbox2["bottom"]))
        gap = max(h_gap, v_gap)
        passed = gap >= min_gap
        detail = f"gap={gap:.0f}px (need >={min_gap}px)"

    elif c.type == "size":
        comp = c.params.get("compare", "larger")
        orig_w = c.params.get("orig_width", 0)
        orig_h = c.params.get("orig_height", 0)
        if comp == "larger":
            passed = bbox["width"] > orig_w or bbox["height"] > orig_h
        elif comp == "smaller":
            passed = bbox["width"] < orig_w or bbox["height"] < orig_h
        detail = f"current={bbox['width']:.0f}x{bbox['height']:.0f}, original={orig_w}x{orig_h}"

    elif c.type == "visible":
        passed = bbox["width"] > 0 and bbox["height"] > 0
        detail = f"size={bbox['width']:.0f}x{bbox['height']:.0f}"

    return {
        "constraint": c.description or f"{c.type}({c.selector})",
        "passed": passed,
        "actual": bbox,
        "detail": detail,
    }


def evaluate_constraints(html: str, constraints: list[Constraint]) -> list[dict]:
    """Sync wrapper for constraint evaluation."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    lambda: asyncio.run(_evaluate_constraints_async(html, constraints))
                ).result(timeout=30)
        else:
            return loop.run_until_complete(_evaluate_constraints_async(html, constraints))
    except RuntimeError:
        return asyncio.run(_evaluate_constraints_async(html, constraints))


# ─── Random World Generator ─────────────────────────────────────────

def _rand_color() -> tuple[str, str]:
    """Return (name, hex)."""
    name = random.choice(list(COLORS.keys()))
    return name, COLORS[name]


def generate_random_world(seed: int | None = None) -> list[UIElement]:
    """Generate a random UI scene with 4-7 elements."""
    if seed is not None:
        random.seed(seed)

    elements = []
    el_id = 0

    # Always add a logo (top-left)
    cname, chex = _rand_color()
    shape = random.choice(LOGO_SHAPES)
    elements.append(UIElement(
        id="logo", el_type="logo",
        x=24, y=24, width=48, height=48,
        content=random.choice(["A", "B", "Z", "X", "◆", "●"]),
        color=chex, font_size=20,
        extra={"shape": shape["type"]},
    ))
    el_id += 1

    # Navigation (top area)
    elements.append(UIElement(
        id="nav", el_type="nav",
        x=100, y=24, width=400, height=48,
        color="#374151",
        extra={"links": random.sample(["Home", "About", "Features", "Pricing", "Blog", "Contact"], 3)},
    ))

    # Title
    _, tc = _rand_color()
    elements.append(UIElement(
        id="title", el_type="title",
        x=80, y=140, width=600, height=60,
        content=random.choice(TITLE_TEXTS),
        color="#1f2937", font_size=random.choice([36, 40, 44]),
    ))

    # Subtitle text
    elements.append(UIElement(
        id="subtitle", el_type="text",
        x=80, y=210, width=500, height=50,
        content="A modern platform for teams who want to move fast.",
        color="#6b7280", font_size=16,
    ))

    # Primary button
    _, bc = _rand_color()
    elements.append(UIElement(
        id="primary-btn", el_type="button",
        x=80, y=290, width=180, height=48,
        content=random.choice(BUTTON_LABELS),
        color=bc, font_size=16,
    ))

    # Secondary button (sometimes)
    if random.random() > 0.4:
        elements.append(UIElement(
            id="secondary-btn", el_type="button",
            x=280, y=290, width=160, height=48,
            content=random.choice(["Learn More", "Watch Demo", "See Docs"]),
            color="#ffffff",
            font_size=16,
            extra={"outline": True},
        ))

    # Cards row (2-3 cards)
    n_cards = random.choice([2, 3])
    card_y = 400
    card_w = 280
    card_gap = 24
    start_x = 80
    for i in range(n_cards):
        _, cc = _rand_color()
        elements.append(UIElement(
            id=f"card-{i}", el_type="card",
            x=start_x + i * (card_w + card_gap), y=card_y,
            width=card_w, height=200,
            content=random.choice(CARD_TITLES),
            color=cc,
            extra={"icon": random.choice(ICON_TYPES)},
        ))

    return elements


# ─── Scenario Generation ────────────────────────────────────────────

def generate_10_scenarios() -> list[UIScenario]:
    """Ten curated spatial/layout evaluation scenarios."""

    scenarios = []

    # ── 1. Move logo to top-right ──
    els = generate_random_world(seed=101)
    scenarios.append(UIScenario(
        "ui_pos_1", els,
        "Move the logo to the top-right corner of the page",
        [Constraint("position", "#logo", {"region": "top-right", "margin": 80},
                    "Logo should be in the top-right corner")],
        "medium", "position",
    ))

    # ── 2. Move button to bottom-center ──
    els = generate_random_world(seed=102)
    scenarios.append(UIScenario(
        "ui_pos_2", els,
        "Move the primary button to the bottom-center of the page",
        [Constraint("position", "#primary-btn", {"region": "bottom-center", "margin": 80},
                    "Button should be at bottom-center")],
        "medium", "position",
    ))

    # ── 3. Center the title horizontally ──
    els = generate_random_world(seed=103)
    scenarios.append(UIScenario(
        "ui_center_1", els,
        "Center the title text horizontally on the page",
        [Constraint("center", "#title", {"axis": "horizontal", "tolerance": 50},
                    "Title should be horizontally centered")],
        "easy", "center",
    ))

    # ── 4. Center the button both axes ──
    els = generate_random_world(seed=104)
    scenarios.append(UIScenario(
        "ui_center_2", els,
        "Center the primary button in the middle of the page (both horizontally and vertically)",
        [Constraint("center", "#primary-btn", {"axis": "both", "tolerance": 60},
                    "Button should be centered on both axes")],
        "medium", "center",
    ))

    # ── 5. Add spacing between cards ──
    els = generate_random_world(seed=105)
    # Squish cards together first
    for el in els:
        if el.id.startswith("card-"):
            idx = int(el.id.split("-")[1])
            el.x = 80 + idx * 260  # tight spacing
    scenarios.append(UIScenario(
        "ui_space_1", els,
        "Add more spacing between the cards — at least 40px gap between each card",
        [Constraint("spacing", "#card-0", {"selector2": "#card-1", "min_gap": 40},
                    "Gap between card 0 and card 1 should be >= 40px")],
        "medium", "spacing",
    ))

    # ── 6. Move nav to bottom ──
    els = generate_random_world(seed=106)
    scenarios.append(UIScenario(
        "ui_pos_3", els,
        "Move the navigation bar to the bottom of the page",
        [Constraint("position", "#nav", {"region": "bottom-center", "margin": 80},
                    "Nav should be at the bottom")],
        "medium", "position",
    ))

    # ── 7. Center logo horizontally ──
    els = generate_random_world(seed=107)
    scenarios.append(UIScenario(
        "ui_center_3", els,
        "Center the logo horizontally at the top of the page",
        [Constraint("center", "#logo", {"axis": "horizontal", "tolerance": 50},
                    "Logo should be horizontally centered"),
         Constraint("position", "#logo", {"region": "top-left", "margin": 800},
                    "Logo should stay near the top")],
        "easy", "center",
    ))

    # ── 8. Move first card to top-right ──
    els = generate_random_world(seed=108)
    scenarios.append(UIScenario(
        "ui_pos_4", els,
        "Move the first card to the top-right area of the page",
        [Constraint("position", "#card-0", {"region": "top-right", "margin": 100},
                    "First card should be in the top-right")],
        "hard", "position",
    ))

    # ── 9. Center all cards vertically ──
    els = generate_random_world(seed=109)
    scenarios.append(UIScenario(
        "ui_center_4", els,
        "Center the row of cards vertically on the page",
        [Constraint("center", "#card-0", {"axis": "vertical", "tolerance": 80},
                    "Cards should be vertically centered")],
        "hard", "center",
    ))

    # ── 10. Swap logo and button positions ──
    els = generate_random_world(seed=110)
    logo_el = next(e for e in els if e.id == "logo")
    btn_el = next(e for e in els if e.id == "primary-btn")
    scenarios.append(UIScenario(
        "ui_swap_1", els,
        f"Swap the positions of the logo and the primary button",
        [Constraint("position", "#logo", {"region": "bottom-left", "margin": 350},
                    "Logo should be roughly where button was"),
         Constraint("position", "#primary-btn", {"region": "top-left", "margin": 100},
                    "Button should be roughly where logo was")],
        "hard", "position",
    ))

    # ── 11. Long-horizon: Multi-step product page layout ──
    product_els = [
        # Logo — placed in body, not in a header
        UIElement(id="logo", el_type="logo",
                  x=40, y=300, width=48, height=48,
                  content="S", color="#6366f1", font_size=20,
                  extra={"shape": "rounded-rect"}),
        # Product image — too large, wrong position
        UIElement(id="product-img", el_type="card",
                  x=30, y=80, width=500, height=350,
                  content="Product Preview", color="#3b82f6",
                  extra={"icon": "globe"}),
        # CTA button — misaligned from image
        UIElement(id="cta-btn", el_type="button",
                  x=700, y=150, width=200, height=48,
                  content="Buy Now", color="#ef4444", font_size=16),
        # Feature cards — uneven spacing
        UIElement(id="card-0", el_type="card",
                  x=30, y=470, width=240, height=160,
                  content="Fast", color="#22c55e", extra={"icon": "bolt"}),
        UIElement(id="card-1", el_type="card",
                  x=290, y=470, width=240, height=160,
                  content="Secure", color="#8b5cf6", extra={"icon": "shield"}),
        UIElement(id="card-2", el_type="card",
                  x=600, y=470, width=240, height=160,
                  content="Global", color="#f97316", extra={"icon": "globe"}),
        # No footer text yet
    ]
    scenarios.append(UIScenario(
        "ui_long_1", product_els,
        "Make these changes to the product page:\n"
        "1. Move the logo to the top-left corner as a header element (top: ~20px, left: ~20px)\n"
        "2. Make the product card smaller (width around 350px, height around 250px)\n"
        "3. Position the CTA button directly to the right of the product card with a small gap\n"
        "4. Make the three feature cards evenly spaced with at least 30px gap between them\n"
        "5. Add a footer text at the bottom-center that says '© 2026 Acme Inc.' (use a <p> tag with id='footer')",
        [
            Constraint("position", "#logo", {"region": "top-left", "margin": 80},
                        "Logo should be in the top-left header area"),
            Constraint("size", "#product-img", {"compare": "smaller", "orig_width": 500, "orig_height": 350},
                        "Product card should be smaller than 500x350"),
            Constraint("spacing", "#card-0", {"selector2": "#card-1", "min_gap": 30},
                        "Gap between feature cards 0 and 1 should be >= 30px"),
            Constraint("spacing", "#card-1", {"selector2": "#card-2", "min_gap": 30},
                        "Gap between feature cards 1 and 2 should be >= 30px"),
            Constraint("visible", "#footer", {},
                        "Footer text element should exist and be visible"),
        ],
        "expert", "long_horizon",
    ))

    return scenarios


# ─── Render Scenario Preview ─────────────────────────────────────────

def render_scenario(scenario: UIScenario) -> list[str]:
    """Render initial scene → [screenshot_b64] as preview."""
    html = render_html(scenario.elements)
    return [screenshot(html)]


# ─── Full Evaluation ────────────────────────────────────────────────

def evaluate_vlm_response(
    scenario: UIScenario,
    vlm_html: str,
) -> dict:
    """
    Evaluate VLM's modified HTML/CSS against the scenario constraints.

    Returns:
        {
            "passed": int,
            "total": int,
            "score": float,  # 0.0 - 1.0
            "before_image": str,  # base64 screenshot
            "after_image": str,   # base64 screenshot of VLM's code
            "constraint_results": [...]
        }
    """
    # Render before
    before_html = render_html(scenario.elements)
    before_img = screenshot(before_html)

    # Render VLM's output
    after_img = screenshot(vlm_html)

    # Check constraints
    results = evaluate_constraints(vlm_html, scenario.constraints)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    return {
        "passed": passed,
        "total": total,
        "score": round(passed / total, 2) if total > 0 else 0.0,
        "before_image": before_img,
        "after_image": after_img,
        "constraint_results": results,
    }


# ─── Get source code for VLM ────────────────────────────────────────

def get_source_code(scenario: UIScenario) -> str:
    """Get the HTML/CSS source code that will be shown to the VLM."""
    return render_html(scenario.elements)


# ─── Review HTML ─────────────────────────────────────────────────────

def generate_review_html(scenarios: list[UIScenario] | None = None) -> str:
    """Generate HTML review page showing all scenarios."""
    if scenarios is None:
        scenarios = generate_10_scenarios()

    cards = []
    for sc in scenarios:
        imgs = render_scenario(sc)
        cards.append(f"""
<div style="background:white;border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.1)">
  <h3 style="margin:0 0 6px;color:#333">{sc.scene_id}</h3>
  <div style="font-size:13px;color:#888;margin-bottom:4px">
    <span style="display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;background:#dbeafe;color:#2563eb">{sc.task_type}</span>
    <span style="display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;background:#f3f4f6;color:#555">{sc.difficulty}</span>
  </div>
  <p style="font-size:14px;color:#333;margin:8px 0 12px">💬 "{sc.instruction}"</p>
  <img src="{imgs[0]}" style="border-radius:8px;border:1px solid #e0e0e0;max-width:100%;max-height:360px" alt="scene">
  <div style="font-size:12px;color:#999;margin-top:8px">Constraints: {', '.join(c.description for c in sc.constraints)}</div>
</div>""")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>UI Eval Scenarios</title></head>
<body style="font-family:-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#f5f5f5">
<h1>🖥️ UI Visual Coding Scenarios</h1>
<p style="color:#666">Spatial layout evaluation · VLM code modification · Playwright rendering</p>
{"".join(cards)}
</body></html>"""


# ─── Long-Horizon Scenario ──────────────────────────────────────────

def _build_product_page() -> list[UIElement]:
    """Build a messy product landing page that needs fixing."""
    return [
        # Logo — wrongly placed in the middle
        UIElement(
            id="logo", el_type="logo",
            x=500, y=300, width=48, height=48,
            content="◆", color="#6366f1", font_size=22,
            extra={"shape": "rounded-rect"},
        ),
        # Header bar (empty — logo should go here)
        UIElement(
            id="header", el_type="nav",
            x=0, y=0, width=1280, height=56,
            color="#1f2937", bg_color="#ffffff",
            extra={"links": ["Products", "Pricing", "Docs"]},
        ),
        # Product image — too small and misaligned
        UIElement(
            id="product-img", el_type="icon",
            x=100, y=120, width=60, height=60,
            color="#3b82f6",
            extra={"icon_type": "globe"},
        ),
        # Product title
        UIElement(
            id="product-title", el_type="title",
            x=100, y=200, width=500, height=50,
            content="SuperWidget Pro", color="#111827", font_size=36,
        ),
        # Product description
        UIElement(
            id="product-desc", el_type="text",
            x=100, y=260, width=500, height=60,
            content="The most powerful widget for modern teams. Boost productivity by 10x with AI-powered automation.",
            color="#6b7280", font_size=15,
        ),
        # CTA button — misaligned (should align with product image)
        UIElement(
            id="cta-btn", el_type="button",
            x=400, y=450, width=200, height=52,
            content="Buy Now — $29", color="#8b5cf6", font_size=16,
        ),
        # Price card
        UIElement(
            id="price-card", el_type="card",
            x=700, y=120, width=320, height=220,
            content="Enterprise", color="#f97316",
            extra={"icon": "shield"},
        ),
        # Footer — no text yet
        UIElement(
            id="footer", el_type="text",
            x=0, y=670, width=1280, height=50,
            content="", color="#9ca3af", font_size=13,
            extra={},
        ),
    ]


def generate_long_horizon_scenario() -> list[UIScenario]:
    """
    One multi-step long-horizon task:
    Fix a messy product page with 5 constraints.
    """
    els = _build_product_page()

    instruction = """Fix this product landing page. Make ALL of the following changes:
1. Move the logo into the header bar (top-left area)
2. Make the product image larger (at least 200x200)
3. Align the CTA button with the product image (same left position)
4. Keep equal vertical spacing between the product title, description, and CTA button
5. Add the text "© 2026 SuperWidget Inc." in the footer"""

    constraints = [
        Constraint("position", "#logo", {"region": "top-left", "margin": 80},
                   "1. Logo should be in the header (top-left)"),
        Constraint("size", "#product-img", {"compare": "larger", "orig_width": 60, "orig_height": 60},
                   "2. Product image should be larger than 60x60"),
        Constraint("position", "#cta-btn", {"region": "top-left", "margin": 800},
                   "3. CTA button left edge should roughly align with content"),
        Constraint("equal_spacing", "", {
            "selectors": ["#product-title", "#product-desc", "#cta-btn"],
            "tolerance": 25,
        }, "4. Equal spacing between title, description, and CTA"),
        Constraint("contains_text", "#footer", {"text": "SuperWidget"},
                   "5. Footer should contain 'SuperWidget' text"),
    ]

    return [UIScenario(
        "ui_long_horizon_1", els,
        instruction,
        constraints,
        "expert", "long_horizon",
    )]
