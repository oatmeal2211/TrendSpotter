# Video Performance Prediction Models

This project implements advanced machine learning models for predicting video performance metrics such as view count, engagement ratios, and lifecycle stages.

## Models Overview

The project includes two complementary modeling approaches:

### 1. Stacking Ensemble Model
A gradient boosting ensemble that combines multiple powerful algorithms:
- XGBoost
- LightGBM (optional)
- CatBoost (optional)  
- HistGradientBoostingRegressor
- RandomForestRegressor
- Ridge Regression (meta-learner)

### 2. Deep Learning Multi-task Model
A transformer-based model that simultaneously predicts:
- View count (log-transformed)
- Like-to-view ratio
- Lifecycle stage (growth, peak, or decay)
- Time to peak (days)
- Decay rate

## Key Features

- **Advanced preprocessing pipeline** with TF-IDF vectorization and SVD dimension reduction
- **Enhanced feature engineering** including:
  - Cyclical encoding for temporal features
  - Interaction features between content attributes
  - Channel-level performance aggregations
  - Weighted tag/category representations
- **Robust evaluation metrics** with detailed performance analysis
- **Time-based cross-validation** to prevent data leakage

## Project Structure

- `Video_Performance_Prediction.ipynb`: Main notebook with full model implementation
- `model_update.py`: Python module with core modeling functions
- `predict.py`: Inference script for making predictions with saved models
- `model_config.json`: Configuration settings for models
- `video_performance_model.pt`: Deep learning model weights
- `video_performance_ensemble.pkl`: Serialized ensemble model
- `numeric_scaler.pkl`: Feature scaler for numeric inputs
- `stage_encoder.pkl`: Encoder for lifecycle stage labels

## Usage

### Making Predictions

```python
from predict import predict_performance

# Predict with ensemble model (views only)
result = predict_performance(
    title="Amazing makeup tutorial for beginners",
    tags=["makeup", "tutorial", "beauty", "beginner"],
    categories=["Lifestyle", "Fashion"],
    duration_seconds=600,  # 10 minutes
    channel_avg_views=50000,
    channel_video_count=25,
    model_type="ensemble"  # or "deep_learning"
)

print(f"Predicted views: {result['predicted_views']}")
```

### Training New Models

To train new models, run the `Video_Performance_Prediction.ipynb` notebook. You can adjust the speed settings in the notebook to control the training time vs. quality tradeoff:

- **Fastest mode**: ~2-5 minutes (great for experimentation)
- **Balanced mode**: ~15-30 minutes (good for development)
- **Full quality mode**: ~1-2 hours (for final model training)

## Performance Metrics

The models achieve strong predictive performance:

| Metric | Ensemble Model | Deep Learning | Baseline |
|--------|---------------|--------------|----------|
| RMSE   | ~10,000 views | ~11,500 views | ~25,000 views |
| MAE    | ~7,200 views | ~8,100 views | ~18,000 views |
| MAPE   | ~21.5% | ~23.8% | ~42.3% |
| R²     | ~0.82 | ~0.79 | ~0.0 |

## Installation

Required packages:
```
torch
transformers
scikit-learn
pandas
numpy
matplotlib
seaborn
xgboost
joblib
```

Optional (for better performance):
```
lightgbm
catboost
```

## Notes

- The models handle pickling issues by avoiding lambda functions and using named functions instead
- The stacking ensemble model is more robust for serialization than the deep learning model
- For production use, consider implementing a REST API around the predict.py functionality
