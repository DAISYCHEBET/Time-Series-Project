"""
Runner script for training and comparing time series models.
"""
import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models import TimeSeriesModels

if __name__ == "__main__":
    print("=" * 60)
    print("SALES FORECASTING - MODEL TRAINING & COMPARISON")
    print("=" * 60)
    
    # Load monthly data with features
    print("\nLoading monthly features data...")
    df = pd.read_csv(
        "data/processed/features_monthly.csv",
        parse_dates=['Date']
    )
    print(f"Loaded {len(df)} months of data")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Initialize models
    print("\nInitializing models...")
    models = TimeSeriesModels()
    
    # Train and compare all models
    print("\nTraining models (this may take a few minutes)...")
    print("-" * 60)
    
    results = models.compare_models(df, forecast_periods=12)
    
    print("\n" + "=" * 60)
    print(" MODEL TRAINING COMPLETED!")
    print("=" * 60)
    print(f"\nSuccessfully trained {len(results)} models:")
    for model_name in results.keys():
        print(f"  ✓ {model_name}")
    
    print("\n" + "=" * 60)
    print(" VIEW RESULTS IN MLFLOW")
    print("=" * 60)
    print("\nTo compare model performance:")
    print("1. Run this command in a NEW terminal:")
    print("   mlflow ui")
    print("\n2. Open your browser to:")
    print("   http://localhost:5000")
    print("\n3. You'll see all models with their metrics (MAE, RMSE, MAPE)")
    print("=" * 60)
    
    # Show sample forecast from Prophet
    if 'Prophet' in results:
        print("\n SAMPLE FORECAST (Prophet - Next 12 Months):")
        forecast = results['Prophet']['forecast']
        future_forecast = forecast.tail(12)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        print(future_forecast.to_string(index=False))