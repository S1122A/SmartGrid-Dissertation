# Smart Grid Sensor Network for Industrial Energy Management

MSc Computer Science research project (COM748), Ulster University.
Saeed Sarwar Anas — Student Number 20068400 — Supervisor: Dr Anwar Haq

A low-cost, end-to-end pipeline for industrial energy management: IoT sensing,
LSTM demand forecasting, unsupervised anomaly detection, and a monitoring
dashboard — built to be affordable enough for a small or medium enterprise.

## Results

Evaluated on two public benchmark datasets.

| Dataset | Model | MAE | RMSE | Improvement vs naive | vs moving average |
|---|---|---|---|---|---|
| UCI Household | LSTM | 0.3707 kW | 0.5421 kW | 8.6% | 35.7% |
| Tetouan City | LSTM | 1,521.60 kW | 2,299.07 kW | 62.0% | 87.0% |

Anomaly detection (UCI): 346 of 34,589 hourly records flagged. Against a
Z-score reference flagging 440, the methods agreed on 267; 79 were found only
by the multivariate detector. These are **agreement** figures, not accuracy —
no ground-truth fault labels exist for either dataset.

Training ran on a laptop CPU (AMD Ryzen 7 6800U, 16 GB RAM) in roughly eleven
minutes. No GPU was used at any stage.

## Repository structure

```
notebooks/     Jupyter notebooks, run in numerical order
data/          Processed hourly datasets (raw data excluded — see Setup)
models/        Trained model artefacts
results/       Figures and CSV outputs consumed by the dashboard
dashboard/     Streamlit monitoring interface
docs/          Research paper and supporting material (LaTeX)
```

## Setup

Requires Python 3.11.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

The raw UCI dataset is not committed (it is large and publicly available).
Download `household_power_consumption.txt` from the UCI Machine Learning
Repository and place it in `data/`.

## Running

Run the notebooks in order:

| Notebook | Purpose | Produces |
|---|---|---|
| `01_data_exploration.ipynb` | Loads raw data, cleans, resamples to hourly, exploratory plots | `data/clean_hourly_power.csv`, EDA figures |
| `02_lstm_model.ipynb` | Trains the LSTM, evaluates against baselines | `results/predictions.csv`, `results/model_comparison.csv` |
| `03_anomaly_detection.ipynb` | Isolation Forest, agreement study vs Z-score | `results/detected_anomalies.csv` |
| `04_tetouan_validation.ipynb` | Re-applies the unchanged pipeline to Tetouan | `results/tetouan_*.csv` |

Then launch the dashboard:

```bash
python -m streamlit run dashboard/app.py
```

It opens at `http://localhost:8501`. Use `python -m streamlit` rather than
`streamlit run` if your system policy blocks executables.

The dashboard reads **pre-computed** results only — nothing trains at run time,
so it starts instantly and cannot fail mid-demonstration.

## Reproducibility

Random seeds are fixed at 42 throughout. The chronological train/validation/
test split (70/15/15) is applied **before** the scaler is fitted, and the
scaler is fitted on the training partition only, so no information from future
time steps enters the model input.

## Datasets

- Individual Household Electric Power Consumption — UCI ML Repository.
  2,075,259 minute-level records, Dec 2006 – Nov 2010, resampled to 34,589
  hourly records.
- Power Consumption of Tetouan City — UCI ML Repository. 52,416 ten-minute
  records for 2017 across three distribution zones, plus weather; resampled to
  8,736 hourly records.

## Scope

The evaluation uses secondary data from the two public benchmarks above. The
IoT sensing layer (Arduino Nano 33 BLE Sense) was designed and specified, with
site permission obtained, but was not deployed — collecting primary data is
identified as future work.

