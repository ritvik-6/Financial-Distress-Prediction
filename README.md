# Hybrid Network Analysis & ML Model for Financial Distress Prediction

A machine learning system that predicts financial distress in companies 
by combining graph-based network analysis with traditional ML classifiers.

## What it does
- Builds financial relationship networks between companies using 
  similarity and correlation of key financial indicators
- Extracts 7 graph features per company (PageRank, centrality, 
  clustering coefficients etc.)
- Combines network features with financial ratios to improve prediction
- Compares 4 ML models: Logistic Regression, Decision Tree, 
  KNN, Passive-Aggressive Classifier

## Tech Stack
- Python, Pandas, NumPy, scikit-learn
- NetworkX (graph construction & analysis)
- Matplotlib, Seaborn (visualization)
- Django (web interface)

## How to Run
```bash
pip install -r requirements.txt
python manage.py runserver
```

## Results
Network-augmented features outperform standalone 
financial ratio models across all classifiers.
