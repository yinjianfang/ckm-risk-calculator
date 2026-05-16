# Advanced CKM Risk Identification Tool

This is a Streamlit web application for identifying advanced CKM risk in patients with diabetic kidney disease using a trained XGBoost model.

## Files

- `app.py`: Streamlit application
- `xgb_model.pkl`: trained XGBoost model
- `requirements.txt`: Python dependencies

## How to run locally

1. Put these files in the same folder.
2. Open a terminal in this folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

5. Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Model inputs

The model expects the following features in this order:

`DR`, `lipid`, `insulin`, `age`, `BUN`, `UACR`, `eGFR`, `TG`, `NEU`, `Hb`

Binary variables are encoded as Yes=1 and No=0.

## Note

This application is intended for research demonstration and should not replace clinical judgment.
