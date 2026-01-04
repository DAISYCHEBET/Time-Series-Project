"""
Find the best XGBoost model from MLflow.
"""
import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("mlruns")

# Get the experiment
client = MlflowClient()
experiment = client.get_experiment_by_name("sales_forecasting")

if experiment:
    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'XGBoost'",
        order_by=["metrics.mae ASC"],
        max_results=1
    )
    
    if runs:
        best_run = runs[0]
        print("=" * 60)
        print(" BEST XGBOOST MODEL FOUND")
        print("=" * 60)
        print(f"Run ID: {best_run.info.run_id}")
        print(f"MAE: {best_run.data.metrics.get('mae', 'N/A')}")
        print(f"RMSE: {best_run.data.metrics.get('rmse', 'N/A')}")
        print(f"MAPE: {best_run.data.metrics.get('mape', 'N/A')}")
        print("=" * 60)
        print(f"\nThis model will be used in the API automatically!")
    else:
        print("No XGBoost runs found. Run: python run_models.py")
else:
    print("Experiment 'sales_forecasting' not found.")