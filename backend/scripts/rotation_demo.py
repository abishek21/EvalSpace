#!/usr/bin/env python3
"""
Demo: DISE Intrinsic-Dynamic Rotation Tasks

Generates a battery of cube rotation tasks, simulates each in MuJoCo,
renders before/after views, and outputs an interactive HTML report.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.dise_tasks import (
    generate_rotation_task,
    simulate_rotation,
    generate_rotation_demo_html,
    FACE_LABELS,
)

def main():
    print("🎲 DISE I-D: Rotation Reasoning Demo")
    print("=" * 50)

    # Define a focused set of tasks
    tasks = [
        generate_rotation_task("Y", 90, "clockwise"),
        generate_rotation_task("Y", 90, "counterclockwise"),
        generate_rotation_task("X", 90, "clockwise"),
        generate_rotation_task("X", 90, "counterclockwise"),
        generate_rotation_task("Z", 90, "clockwise"),
        generate_rotation_task("Z", 180, "clockwise"),
    ]

    results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task.angle_deg}° {task.direction} around {task.axis}-axis")
        print(f"  Question: {task.question[:80]}...")
        print(f"  Ground truth: {task.ground_truth} → {FACE_LABELS[task.ground_truth]}")

        result = simulate_rotation(task, render_views=True)
        results.append(result)

        status = "✅" if result.correct else "❌"
        print(f"  MuJoCo result: {result.predicted_top} → {FACE_LABELS[result.predicted_top]} {status}")

    # Summary
    correct = sum(1 for r in results if r.correct)
    print(f"\n{'=' * 50}")
    print(f"Results: {correct}/{len(results)} verified correct")

    # Generate HTML
    html = generate_rotation_demo_html(results)
    out_path = os.path.join(os.path.dirname(__file__), "..", "rotation_demo.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\n📄 HTML report: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
