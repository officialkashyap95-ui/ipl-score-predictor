# 🏏 IPL Score Predictor

ML-powered app to predict first-innings IPL scores using real ball-by-ball data (2008–2025).

## 📁 Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `generate_and_train.py` | Train the model |
| `model.pkl` | Saved ML model |
| `encoders.pkl` | Label encoders |
| `meta.json` | Teams & venues list |
| `requirements.txt` | Dependencies |

## 🚀 How to Run

### Step 1 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 – Place CSV files in the same folder
- `deliveries.csv`
- `matches.csv`

### Step 3 – Train the model (run once)
```bash
python generate_and_train.py
```

### Step 4 – Launch the app
```bash
streamlit run app.py
```
Then open **http://localhost:8501**

## 🤖 Model Details
- **Algorithm:** Random Forest Regressor
- **R² Score:** ~80.3%
- **Training Data:** 134,190 deliveries from 1,158 matches
- **Features:** Batting team, bowling team, venue, current over, cumulative score, wickets fallen, run rate, last-5-overs runs

## 🌐 Deploy Free Online
1. Push files to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → Deploy!
