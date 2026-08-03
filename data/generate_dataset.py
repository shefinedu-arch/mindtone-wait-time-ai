"""
Synthetic outpatient dataset generator — Mindtone Mental Health Clinic
Simulates ~6 months of outpatient visits with realistic relationships
between queue load, staffing, appointment type and wait time, so a
regression model has real signal to learn. (Internally simulates a second
clinic too, for realistic shared scheduling dynamics, then filters to
Mindtone-only for this project's dataset.)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
N = 6000

clinics = ["Revive Skin Clinic", "Mindtone Mental Health Clinic"]
departments = {
    "Revive Skin Clinic": ["Dermatology", "Cosmetic Procedures", "Skin Allergy"],
    "Mindtone Mental Health Clinic": ["Psychiatry", "Counseling", "Addiction Therapy"],
}
doctors_per_dept = 3
booking_channels = ["Online", "Phone", "Walk-in"]
appointment_types = ["Scheduled - New", "Scheduled - Follow-up", "Walk-in"]

start_date = datetime(2025, 1, 1)

rows = []
for i in range(N):
    clinic = rng.choice(clinics)
    dept = rng.choice(departments[clinic])
    doctor_id = f"{dept[:3].upper()}-D{rng.integers(1, doctors_per_dept+1)}"
    doctor_experience = int(rng.integers(1, 26))

    visit_date = start_date + timedelta(days=int(rng.integers(0, 182)))
    day_of_week = visit_date.strftime("%A")
    is_weekend = day_of_week in ["Saturday", "Sunday"]

    # clinic operating hours 9am-6pm
    appt_hour = int(rng.integers(9, 18))
    appt_minute = int(rng.choice([0, 15, 30, 45]))
    appointment_time = f"{appt_hour:02d}:{appt_minute:02d}"

    appt_type = rng.choice(appointment_types, p=[0.35, 0.45, 0.20])
    booking_channel = "Walk-in" if appt_type == "Walk-in" else rng.choice(["Online", "Phone"], p=[0.7, 0.3])

    patient_age = int(np.clip(rng.normal(38, 14), 5, 85))
    is_first_visit = 1 if appt_type == "Scheduled - New" else int(rng.random() < 0.1)

    # operational factors
    staff_on_duty = int(rng.integers(2, 7))              # front-desk + nurses on duty
    doctors_on_duty = int(rng.integers(1, 4))
    patients_ahead = int(np.clip(rng.poisson(5 if not is_weekend else 8), 0, 25))
    # peak-hour effect: lunch (12-14) and evening (16-18) busier
    peak_factor = 1.4 if appt_hour in [12, 13, 16, 17] else 1.0
    # walk-ins add unpredictability to queue
    walkin_load = int(rng.poisson(3)) if appt_type == "Walk-in" else int(rng.poisson(1))

    avg_service_time = float(np.clip(rng.normal(18 if dept in ["Psychiatry", "Counseling", "Addiction Therapy"] else 12, 4), 6, 40))

    # ----- target: actual wait time (minutes) -----
    base = 8
    queue_effect = patients_ahead * (avg_service_time / max(doctors_on_duty, 1)) * 0.28
    staffing_relief = staff_on_duty * 1.6
    walkin_penalty = walkin_load * 3.2 if appt_type == "Walk-in" else walkin_load * 1.1
    peak_effect = (peak_factor - 1) * 22
    weekend_effect = 6 if is_weekend else 0
    experience_relief = (doctor_experience / 25) * 4  # experienced doctors slightly faster turnover

    noise = rng.normal(0, 6)
    wait_time = base + queue_effect + walkin_penalty + peak_effect + weekend_effect - staffing_relief - experience_relief + noise
    wait_time = float(np.clip(wait_time, 2, 180))

    consultation_duration = float(np.clip(rng.normal(avg_service_time, 3), 5, 60))

    rows.append({
        "visit_id": f"V{100000+i}",
        "clinic": clinic,
        "department": dept,
        "doctor_id": doctor_id,
        "doctor_experience_years": doctor_experience,
        "visit_date": visit_date.strftime("%Y-%m-%d"),
        "day_of_week": day_of_week,
        "is_weekend": int(is_weekend),
        "appointment_time": appointment_time,
        "appointment_hour": appt_hour,
        "appointment_type": appt_type,
        "booking_channel": booking_channel,
        "patient_age": patient_age,
        "is_first_visit": is_first_visit,
        "staff_on_duty": staff_on_duty,
        "doctors_on_duty": doctors_on_duty,
        "patients_ahead_in_queue": patients_ahead,
        "walkin_load": walkin_load,
        "avg_service_time_doctor_min": round(avg_service_time, 1),
        "consultation_duration_min": round(consultation_duration, 1),
        "actual_wait_time_min": round(wait_time, 1),
    })

df = pd.DataFrame(rows)
mindtone = df[df["clinic"] == "Mindtone Mental Health Clinic"].drop(columns=["clinic"]).reset_index(drop=True)
mindtone.to_csv("data/mindtone_wait_times.csv", index=False)
print(mindtone.shape)
print(mindtone.head())
print(mindtone["actual_wait_time_min"].describe())
