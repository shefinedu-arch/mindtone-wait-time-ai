# Mindtone Mental Health Clinic — Outpatient Wait-Time AI Prototype

## Structure
- `data/generate_dataset.py` — generates the synthetic outpatient dataset (produces `data/outpatient_wait_times.csv`)
- `data/mindtone_wait_times.csv` — Mindtone-only dataset used for this project (2,976 visits, 3 departments)
- `train_model_mindtone.py` — trains & compares Gradient Boosting vs Random Forest, saves model + evaluation charts
- `model/simulate_impact.py` — simulates AI-optimized scheduling vs current process to project wait-time reduction
- `model/architecture_diagram.py` — generates the system architecture diagram
- `model/timeline_chart.py` — generates the development timeline (Gantt) chart
- `model/wait_time_model.joblib` — trained Gradient Boosting pipeline (best model)
- `results/` — contains evaluation metrics (`metrics.json`) and plot visualizations (`.png`)

## Run Order
```powershell
python train_model_mindtone.py     # Trains model on data/mindtone_wait_times.csv & outputs results
python model/simulate_impact.py     # Runs wait-time reduction simulation
python model/architecture_diagram.py
python model/timeline_chart.py