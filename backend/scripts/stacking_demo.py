#!/usr/bin/env python3
"""
Demo: DISE Stacking Stability — 10 scenarios verified by MuJoCo physics.

MuJoCo is the SOLE source of ground truth:
  Build stack → simulate 3s → did it topple? That's the answer.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.dise_stacking import (
    generate_10_scenarios,
    simulate_stacking,
    generate_stacking_demo_html,
    STACKABLE_OBJECTS,
)


def main():
    print("📦 DISE E-D: Stacking Stability Demo")
    print("=" * 55)
    print("MuJoCo physics decides ground truth. No shortcuts.\n")

    scenarios = generate_10_scenarios()
    results = []

    for i, scenario in enumerate(scenarios):
        print(f"[{i+1}/10] {scenario.name}")
        obj_names = [f"{STACKABLE_OBJECTS[o.obj_type]['label']}({o.color})" for o in scenario.objects]
        print(f"  Stack: {' → '.join(obj_names)}")

        t0 = time.time()
        result = simulate_stacking(scenario, settle_seconds=3.0)
        elapsed = time.time() - t0

        if result.stable:
            print(f"  ✅ STABLE (max disp: {result.max_displacement*100:.1f}cm) [{elapsed:.1f}s]")
        else:
            fell = [n.split("_", 2)[-1] for n in result.fell_objects]
            print(f"  ❌ UNSTABLE — fell: {', '.join(fell)} (max disp: {result.max_displacement*100:.1f}cm) [{elapsed:.1f}s]")

        results.append(result)

    # Summary
    n_stable = sum(1 for r in results if r.stable)
    print(f"\n{'=' * 55}")
    print(f"Results: {n_stable} stable, {10 - n_stable} unstable")

    # Generate HTML
    html = generate_stacking_demo_html(results)
    out = os.path.join(os.path.dirname(__file__), "..", "stacking_demo.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"📄 HTML report: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
