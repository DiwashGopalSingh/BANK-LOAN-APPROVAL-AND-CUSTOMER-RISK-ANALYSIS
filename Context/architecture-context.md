# Architecture Context

## Stack
- **Language:** Python 3.x
- **Data mining:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib, seaborn
- **UI:** Streamlit
- **Model persistence:** joblib (`.pkl` files)

## Folder Structure
```
bank-loan-risk-project/
├── context/                  # This folder — agent context files
├── data/
│   ├── raw/                  # Original dataset, untouched
│   └── processed/            # Cleaned/engineered data
├── notebooks/                # EDA and experimentation (.ipynb)
├── src/
│   ├── preprocessing.py      # Cleaning, encoding, feature engineering
│   ├── train_approval.py     # Trains loan approval classifier(s)
│   ├── train_risk.py         # Builds risk scoring/segmentation
│   └── evaluate.py           # Metrics, comparison tables
├── models/                   # Saved trained models (.pkl)
├── app.py                    # Streamlit app entry point
├── report/                   # Final written report
└── requirements.txt
```

## Data Flow
1. `data/raw/` → loaded and explored in `notebooks/`
2. `src/preprocessing.py` cleans and transforms → `data/processed/`
3. `src/train_approval.py` and `src/train_risk.py` train models on processed data → save to `models/`
4. `app.py` loads the saved models and serves predictions through the Streamlit UI

## Key Design Decisions
- Keep approval classification and risk scoring as **separate, independent components** — the app calls both, but neither depends on the other's internals. This keeps debugging simpler.
- Models are trained offline (via scripts/notebooks) and loaded pre-trained into the app — the app itself never retrains a model live.
- No database — the app is a single-session tool; a submitted form doesn't need to persist between runs.
