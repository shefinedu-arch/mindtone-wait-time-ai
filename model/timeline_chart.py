import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

phases = [
    ("1. Discovery & requirements", 0, 2, "#185FA5"),
    ("2. Data collection & integration", 1.5, 3, "#0F6E56"),
    ("3. Model development (wait-time engine)", 3.5, 3, "#993C1D"),
    ("4. Queue optimizer & scheduling logic", 4.5, 3, "#993C1D"),
    ("5. Dashboard & patient app (UI)", 5, 3.5, "#534AB7"),
    ("6. Integration & pilot (Psychiatry dept.)", 8, 2.5, "#888780"),
    ("7. Staff training & feedback loop", 9, 2, "#888780"),
    ("8. Full rollout (all departments)", 10.5, 1.5, "#639922"),
    ("9. Monitoring & optimization", 12, 2, "#639922"),
]

fig, ax = plt.subplots(figsize=(11, 5.5))
for i, (name, start, dur, color) in enumerate(phases):
    y = len(phases) - i
    ax.barh(y, dur, left=start, height=0.55, color=color, alpha=0.85)
    ax.text(start + dur/2, y, f"{dur:.1f} wk", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

ax.set_yticks(range(len(phases), 0, -1))
ax.set_yticklabels([p[0] for p in phases], fontsize=9.5)
ax.set_xlabel("Weeks from project start")
ax.set_xlim(0, 14.5)
ax.set_xticks(range(0, 15, 1))
ax.set_title("Development Timeline — 14 Weeks (~3.5 months)", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("results/timeline_gantt.png", dpi=160, facecolor="white")
print("saved timeline chart")
