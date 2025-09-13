# L'Oréal Datathon: Video Trends & View Prediction

Analyze YouTube beauty content to uncover trends (emerging vs. established), optimize video duration, and predict video views using a saved ML pipeline.

## Repository Contents

- `videos.csv` — Source dataset of videos used in all notebooks.
- `Loreal_Cleaning.ipynb` — Data loading, cleaning, normalization, and feature extraction helpers (duration parsing, tag merges, topic extraction from Wikipedia URLs).
- `Loreal_Emerging.ipynb` — Analyses to discover emerging topics/tags and entry/decay detection logic.
- `Loreal_Established.ipynb` — End‑to‑end analysis for established trends, duration optimization, and the ML pipeline to predict views. Also includes a test cell to run predictions with the saved model.
- `Loreal_Model.ipynb` — Additional modeling and experiments (EDA + model variants and visualizations).
- `Trend_Decay.ipynb` — Trend lifecycle exploration (entry/growth/decay heuristics, category and tag time‑series views).
- `video_views_predictor.joblib` — Trained scikit‑learn Pipeline for predicting video views (target trained on log(viewCount)).
- `README.md` — This guide.

## What the Project Does

1. Cleans and enriches video metadata (duration in seconds, merged tags, topics extracted from Wikipedia URLs).
2. Explores trends by category and tag over time to separate emerging vs. established themes.
3. Optimizes video duration via bucketed analytics and efficiency metrics (views per second).
4. Trains a text+numeric feature ML pipeline and saves it as `video_views_predictor.joblib` for inference.

## Environment Setup (Windows, PowerShell)

Install Python 3.11+ and the required packages:

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost nltk joblib wikipedia-api requests
```

First‑time NLTK use will download stopwords automatically in the notebook. If it doesn’t, run:

```
python -c "import nltk; nltk.download('stopwords')"
```

## Using the Saved Model (Quick Start)

You can load and run predictions directly in a notebook or a small script. The model expects a DataFrame with the engineered fields used during training (clean text fields plus numeric features). Example (matches the testing cell included at the bottom of `Loreal_Established.ipynb`):

```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load('video_views_predictor.joblib')

sample = {
    'title_clean': 'new makeup tutorial beauty tips',
    'tags_clean': 'makeup tutorial beauty tips cosmetics',
    'categories_clean': 'Physical attractiveness Lifestyle',
    'contentDuration_seconds': 60,
    'duration_minutes': 1.0,
    'duration_log': np.log1p(60),
    'publish_hour': 12,
    'publish_day': 5,
    'publish_month': 9,
    'publish_year': 2025,
    'title_length': 31,
    'title_word_count': 5,
    'has_question': 0,
    'has_exclaim': 0,
    'tag_count': 5,
    'topic_count': 2,
    'channel_avg_views': 5000,
    'channel_video_count': 50
}

df = pd.DataFrame([sample])
log_pred = model.predict(df)[0]
predicted_views = np.expm1(log_pred)
print(int(predicted_views))
```

Notes:
- The pipeline predicts log(viewCount). Convert back with `np.expm1`.
- Text columns (`title_clean`, `tags_clean`, `categories_clean`) should already be lower‑cased and cleaned like in the notebooks.

## How the Model Was Built (Short)

- Cleaned inputs from `videos.csv` and created engineered features:
  - Text: TF‑IDF of title, tags, categories with SVD for dimensionality reduction.
  - Numeric: duration features (seconds, minutes, log), publish time (hour, weekday, month, year), title length/word count, tag/topic counts, and channel aggregates.
- Target: `log1p(viewCount_capped)` to reduce skew and cap extreme outliers.
- Models evaluated: Ridge, Gradient Boosting, XGBoost within a unified Pipeline.
- Best pipeline saved to `video_views_predictor.joblib` via joblib.

## Reproducing Notebooks

Run the notebooks in this order for a smooth experience:

1) `Loreal_Cleaning.ipynb` → parsing, cleaning, and feature preparation
2) `Loreal_Established.ipynb` → established trend analysis + model train & save
3) `Loreal_Emerging.ipynb` and `Trend_Decay.ipynb` → complementary trend views
4) `Loreal_Model.ipynb` → additional modeling explorations

If a Wikipedia call fails (rate limits/connectivity), the code falls back gracefully for pageviews; analyses still run.

## Troubleshooting

- ImportError when loading the model: ensure scikit‑learn and xgboost versions are installed as above.
- NLTK stopwords error: run the one‑liner download command shown in Environment Setup.
- Prediction errors about missing columns: build your input DataFrame with the full set of expected fields shown in the example above.
- Large figures not showing: rerun the plotting cell or clear outputs and re‑execute the notebook section.

## License

For Datathon/demo use. Add a license if you plan to open‑source or distribute.

