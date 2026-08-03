import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")

COLORS = {
    "channel": "#E6F1FB", "channel_border": "#185FA5", "channel_text": "#0C447C",
    "core": "#E1F5EE", "core_border": "#0F6E56", "core_text": "#085041",
    "ml": "#FAECE7", "ml_border": "#993C1D", "ml_text": "#712B13",
    "data": "#F1EFE8", "data_border": "#5F5E5A", "data_text": "#2C2C2A",
    "out": "#EEEDFE", "out_border": "#534AB7", "out_text": "#3C3489",
}

def box(x, y, w, h, title, subtitle, fill, border, textcolor, fontsize=10.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=1.3, edgecolor=border, facecolor=fill)
    ax.add_patch(b)
    ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor)
    if subtitle:
        ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=8.7,
                color=textcolor, wrap=True)

def arrow(x1, y1, x2, y2, color="#666"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.2, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# ---- Tier 1: Patient-facing channels ----
ax.text(0.3, 7.6, "Patient & staff access", fontsize=10, fontweight="bold", color="#444")
box(0.3, 6.5, 2.6, 0.9, "Patient web / app", "Book, view live wait", COLORS["channel"], COLORS["channel_border"], COLORS["channel_text"])
box(3.1, 6.5, 2.6, 0.9, "Clinic kiosk / front desk", "Check-in, walk-ins", COLORS["channel"], COLORS["channel_border"], COLORS["channel_text"])
box(5.9, 6.5, 2.6, 0.9, "SMS / WhatsApp alerts", "Wait updates, reminders", COLORS["channel"], COLORS["channel_border"], COLORS["channel_text"])
box(8.7, 6.5, 3.6, 0.9, "Staff / admin dashboard", "Live queue, analytics", COLORS["out"], COLORS["out_border"], COLORS["out_text"])

# ---- Tier 2: API / integration layer ----
box(0.3, 5.15, 12.0, 0.7, "API gateway & authentication layer", "Secure REST/HTTPS, role-based access (patients, staff, admin)",
    "#F5F4EF", "#888780", "#2C2C2A", fontsize=10)

arrow(1.6, 6.5, 2.5, 5.85)
arrow(4.4, 6.5, 4.9, 5.85)
arrow(7.2, 6.5, 7.3, 5.85)
arrow(10.5, 6.5, 10.0, 5.85)

# ---- Tier 3: Core application services ----
ax.text(0.3, 4.75, "Core application services", fontsize=10, fontweight="bold", color="#444")
box(0.3, 3.55, 2.9, 1.0, "Appointment scheduler", "Slot allocation & rebooking", COLORS["core"], COLORS["core_border"], COLORS["core_text"])
box(3.4, 3.55, 2.9, 1.0, "Wait-time prediction engine", "ML inference API (real time)", COLORS["core"], COLORS["core_border"], COLORS["core_text"])
box(6.5, 3.55, 2.9, 1.0, "Queue & load optimizer", "Rebalances doctors/slots", COLORS["core"], COLORS["core_border"], COLORS["core_text"])
box(9.6, 3.55, 2.7, 1.0, "Notification service", "SMS/app push triggers", COLORS["core"], COLORS["core_border"], COLORS["core_text"])

arrow(2.0, 5.15, 1.7, 4.55)
arrow(5.0, 5.15, 4.85, 4.55)
arrow(8.0, 5.15, 8.0, 4.55)
arrow(11.0, 5.15, 11.0, 4.55)

# cross-links among core services
arrow(3.2, 4.05, 3.4, 4.05, color="#0F6E56")
arrow(6.3, 4.05, 6.5, 4.05, color="#0F6E56")
arrow(9.4, 4.05, 9.6, 4.05, color="#0F6E56")

# ---- Tier 4: ML pipeline ----
ax.text(0.3, 3.15, "ML pipeline (offline / scheduled)", fontsize=10, fontweight="bold", color="#444")
box(0.3, 1.95, 2.9, 1.0, "Data ingestion & ETL", "Clinic DB + EHR feeds", COLORS["ml"], COLORS["ml_border"], COLORS["ml_text"])
box(3.4, 1.95, 2.9, 1.0, "Model training & validation", "GradientBoosting wait model", COLORS["ml"], COLORS["ml_border"], COLORS["ml_text"])
box(6.5, 1.95, 2.9, 1.0, "Model registry & monitoring", "Versioning, drift checks", COLORS["ml"], COLORS["ml_border"], COLORS["ml_text"])
box(9.6, 1.95, 2.7, 1.0, "Analytics & reporting", "KPI dashboards, exports", COLORS["ml"], COLORS["ml_border"], COLORS["ml_text"])

arrow(1.7, 3.55, 1.7, 2.95)
arrow(4.85, 3.55, 4.85, 2.95)
arrow(1.7, 2.95, 3.4, 2.45, color="#993C1D")
arrow(6.3, 2.45, 6.5, 2.45, color="#993C1D")
arrow(9.4, 2.45, 9.6, 2.45, color="#993C1D")
arrow(8.0, 2.95, 8.0, 3.55, color="#993C1D")  # model registry serves prediction engine

# ---- Tier 5: Data layer ----
ax.text(0.3, 1.55, "Data layer", fontsize=10, fontweight="bold", color="#444")
box(0.3, 0.3, 3.9, 1.0, "Clinic operational DB", "Appointments, patients, staffing (Mindtone)", COLORS["data"], COLORS["data_border"], COLORS["data_text"], fontsize=10)
box(4.5, 0.3, 3.9, 1.0, "Historical visit data warehouse", "6+ months wait-time & queue history", COLORS["data"], COLORS["data_border"], COLORS["data_text"], fontsize=10)
box(8.7, 0.3, 3.6, 1.0, "Compliance & audit log", "HIPAA-aligned access logs (mental health data)", COLORS["data"], COLORS["data_border"], COLORS["data_text"], fontsize=10)

arrow(1.7, 1.95, 1.7, 1.3)
arrow(6.4, 1.95, 6.4, 1.3)
arrow(10.5, 1.95, 10.5, 1.3)

plt.tight_layout()
plt.savefig("results/architecture_diagram.png", dpi=170, bbox_inches="tight", facecolor="white")
print("saved architecture diagram")
