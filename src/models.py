"""
Time series forecasting models module.
Implements Prophet, SARIMA, and XGBoost with MLflow tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Prophet
from prophet import Prophet

# Statistical models
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# ML model
from xgboost import XGBRegressor

# MLflow for tracking
import mlflow
import mlflow.sklearn

from src.utils import logger, load_config


class TimeSeriesModels:
    """
    Wrapper class for multiple time series forecasting models.
    Includes Prophet, SARIMA, and XGBoost with MLflow tracking.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the TimeSeriesModels.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.model_config = self.config['models']
        self.training_config = self.config['training']
        self.mlflow_config = self.config['mlflow']
        
        # Set MLflow tracking
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
        
        logger.info("TimeSeriesModels initialized")
    
    def prepare_prophet_data(self, df: pd.DataFrame, date_col: str = 'Date', 
                            target_col: str = 'Sales') -> pd.DataFrame:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns).
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            target_col: Target column name
        
        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Apply log transformation if data is skewed
        if prophet_df['y'].skew() > 1:
            prophet_df['y'] = np.log1p(prophet_df['y'])
            logger.info("Applied log transformation for Prophet")
        
        return prophet_df
    
    def train_prophet(self, df: pd.DataFrame, forecast_periods: int = 12) -> Tuple[Prophet, pd.DataFrame]:
        """
        Train Facebook Prophet model.
        
        Args:
            df: DataFrame with 'ds' and 'y' columns
            forecast_periods: Number of periods to forecast
        
        Returns:
            Tuple of (trained_model, forecast_dataframe)
        """
        logger.info("Training Prophet model")
        
        with mlflow.start_run(run_name="Prophet"):
            # Log parameters
            mlflow.log_params({
                "model_type": "Prophet",
                "forecast_periods": forecast_periods,
                **self.model_config['prophet']
            })
            
            # Create and train model
            model = Prophet(
                seasonality_mode=self.model_config['prophet']['seasonality_mode'],
                yearly_seasonality=self.model_config['prophet']['yearly_seasonality'],
                weekly_seasonality=self.model_config['prophet']['weekly_seasonality'],
                daily_seasonality=self.model_config['prophet']['daily_seasonality']
            )
            
            model.fit(df)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=forecast_periods, freq='M')
            forecast = model.predict(future)
            
            # Inverse log transform if applied
            if df['y'].max() < 20:  # Likely log-transformed
                forecast['yhat'] = np.expm1(forecast['yhat'])
                forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
                forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
            
            # Calculate metrics on training data
            train_predictions = forecast.iloc[:len(df)]['yhat'].values
            actual = df['y'].values
            
            # Reverse log if needed
            if df['y'].max() < 20:
                actual = np.expm1(actual)
            
            metrics = self._calculate_metrics(actual, train_predictions)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Prophet trained. MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
        
        return model, forecast
    
    def train_sarima(self, df: pd.DataFrame, forecast_periods: int = 12) -> Tuple[Any, pd.DataFrame]:
        """
        Train SARIMA model with auto parameter selection.
        
        Args:
            df: DataFrame with time series data
            forecast_periods: Number of periods to forecast
        
        Returns:
            Tuple of (trained_model, forecast_dataframe)
        """
        logger.info("Training SARIMA model (this may take a few minutes)...")
        
        with mlflow.start_run(run_name="SARIMA"):
            # Prepare data
            y = df['y'].values if 'y' in df.columns else df['Sales'].values
            
            # Auto ARIMA to find best parameters
            logger.info("Finding optimal SARIMA parameters...")
            auto_model = pm.auto_arima(
                y,
                seasonal=True,
                m=12,  # Monthly seasonality
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            order = auto_model.order
            seasonal_order = auto_model.seasonal_order
            
            logger.info(f"Optimal parameters - Order: {order}, Seasonal: {seasonal_order}")
            
            # Log parameters
            mlflow.log_params({
                "model_type": "SARIMA",
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "forecast_periods": forecast_periods
            })
            
            # Train SARIMA
            model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            
            # Make predictions
            train_predictions = fitted_model.fittedvalues
            forecast_values = fitted_model.forecast(steps=forecast_periods)
            
            # Create forecast dataframe
            last_date = df['ds'].iloc[-1] if 'ds' in df.columns else df['Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq='M')[1:]
            
            forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_values
            })
            
            # Calculate metrics
            metrics = self._calculate_metrics(y, train_predictions)
            mlflow.log_metrics(metrics)
            
            logger.info(f"SARIMA trained. MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
        
        return fitted_model, forecast
    
    def train_xgboost(self, df: pd.DataFrame, forecast_periods: int = 12) -> Tuple[XGBRegressor, pd.DataFrame]:
        """
        Train XGBoost model for time series.
        
        Args:
            df: DataFrame with features
            forecast_periods: Number of periods to forecast
        
        Returns:
            Tuple of (trained_model, forecast_dataframe)
        """
        logger.info("Training XGBoost model")
        
        with mlflow.start_run(run_name="XGBoost"):
            # Prepare features and target
            feature_cols = [col for col in df.columns if col not in ['ds', 'Date', 'y', 'Sales']]
            
            if len(feature_cols) == 0:
                logger.warning("No features found for XGBoost. Creating basic time features...")
                df['month'] = pd.to_datetime(df['ds'] if 'ds' in df.columns else df['Date']).dt.month
                df['year'] = pd.to_datetime(df['ds'] if 'ds' in df.columns else df['Date']).dt.year
                feature_cols = ['month', 'year']
            
            X = df[feature_cols].fillna(0)
            y = df['y'].values if 'y' in df.columns else df['Sales'].values
            
            # Log parameters
            mlflow.log_params({
                "model_type": "XGBoost",
                "n_features": len(feature_cols),
                "forecast_periods": forecast_periods,
                **self.model_config['xgboost']
            })
            
            # Train model
            model = XGBRegressor(
                n_estimators=self.model_config['xgboost']['n_estimators'],
                max_depth=self.model_config['xgboost']['max_depth'],
                learning_rate=self.model_config['xgboost']['learning_rate'],
                random_state=self.training_config['random_state']
            )
            
            model.fit(X, y)
            
            # Predictions on training data
            train_predictions = model.predict(X)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y, train_predictions)
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"XGBoost trained. MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
            
            # Note: Forecasting with XGBoost requires future features
            # This is a simplified version - in production, you'd need to create future features
            forecast = pd.DataFrame({
                'ds': pd.date_range(
                    start=df['ds'].iloc[-1] if 'ds' in df.columns else df['Date'].iloc[-1],
                    periods=forecast_periods + 1,
                    freq='M'
                )[1:],
                'yhat': [y.mean()] * forecast_periods  # Placeholder - needs proper implementation
            })
        
        return model, forecast
    
    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    def compare_models(self, df: pd.DataFrame, forecast_periods: int = 12) -> Dict[str, Any]:
        """
        Train and compare all models.
        
        Args:
            df: Input DataFrame
            forecast_periods: Number of periods to forecast
        
        Returns:
            Dictionary with all models and their forecasts
        """
        logger.info("=" * 50)
        logger.info("COMPARING ALL MODELS")
        logger.info("=" * 50)
        
        results = {}
        
        # Prepare data for Prophet
        prophet_df = self.prepare_prophet_data(df)
        
        # Train Prophet
        try:
            prophet_model, prophet_forecast = self.train_prophet(prophet_df, forecast_periods)
            results['Prophet'] = {
                'model': prophet_model,
                'forecast': prophet_forecast
            }
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
        
        # Train SARIMA
        try:
            sarima_model, sarima_forecast = self.train_sarima(prophet_df, forecast_periods)
            results['SARIMA'] = {
                'model': sarima_model,
                'forecast': sarima_forecast
            }
        except Exception as e:
            logger.error(f"SARIMA training failed: {e}")
        
        # Train XGBoost
        try:
            xgb_model, xgb_forecast = self.train_xgboost(df, forecast_periods)
            results['XGBoost'] = {
                'model': xgb_model,
                'forecast': xgb_forecast
            }
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
        
        logger.info("=" * 50)
        logger.info(f"Model comparison complete. {len(results)} models trained.")
        logger.info("=" * 50)
        
        return results


if __name__ == "__main__":
    # Test the models
    import pandas as pd
    
    # Load monthly data
    df = pd.read_csv("data/processed/features_monthly.csv", parse_dates=['Date'])
    
    # Initialize models
    models = TimeSeriesModels()
    
    # Compare all models
    results = models.compare_models(df, forecast_periods=12)
    
    print("\n Model training completed!")
    print(f"Trained {len(results)} models")
    print("\nCheck MLflow UI to compare results:")
    print("Run: mlflow ui")
    print("Then open: http://localhost:5000")