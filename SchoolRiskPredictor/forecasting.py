import pandas as pd
import numpy as np
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')


class DropoutForecaster:
    """
    Time-series forecasting module for predicting future dropout risk
    Uses Prophet for robust time-series predictions
    """

    def __init__(self):
        self.models = {}
        self.forecasts = {}

    def prepare_time_series_data(self, df, aggregation_level='national'):
        """
        Prepare time-series data for forecasting

        Parameters:
        - df: DataFrame with historical school data
        - aggregation_level: 'national', 'state', or 'district'

        Returns:
        - Dictionary of time series data ready for Prophet
        """
        ts_data = {}

        # Use 'predicted_risk_label' if available (from model), else fall back to 'dropout_risk' (from data)
        risk_col = 'predicted_risk_label' if 'predicted_risk_label' in df.columns else 'dropout_risk'

        if aggregation_level == 'national':
            national_ts = df.groupby('year').agg({
                risk_col:
                lambda x: (x == 'High').sum() / len(x) * 100
                if len(x) > 0 else 0
            }).reset_index()
            national_ts.columns = ['ds', 'y']
            national_ts['ds'] = pd.to_datetime(national_ts['ds'].astype(str) +
                                               '-01-01')
            ts_data['National'] = national_ts

        elif aggregation_level == 'state':
            for state in df['state'].unique():
                state_df = df[df['state'] == state]
                state_ts = state_df.groupby('year').agg({
                    risk_col:
                    lambda x: (x == 'High').sum() / len(x) * 100
                    if len(x) > 0 else 0
                }).reset_index()

                if len(state_ts
                       ) >= 2:  # Prophet requires at least 2 data points
                    state_ts.columns = ['ds', 'y']
                    state_ts['ds'] = pd.to_datetime(
                        state_ts['ds'].astype(str) + '-01-01')
                    ts_data[state] = state_ts

        elif aggregation_level == 'district':
            for district in df['district'].unique():
                district_df = df[df['district'] == district]
                district_ts = district_df.groupby('year').agg({
                    risk_col:
                    lambda x: (x == 'High').sum() / len(x) * 100
                    if len(x) > 0 else 0
                }).reset_index()

                if len(district_ts) >= 2:
                    district_ts.columns = ['ds', 'y']
                    district_ts['ds'] = pd.to_datetime(
                        district_ts['ds'].astype(str) + '-01-01')
                    ts_data[district] = district_ts

        return ts_data

    def forecast_dropout_risk(self,
                              ts_data,
                              periods=3,
                              region_name='National'):
        """
        Forecast dropout risk for future periods using Prophet
        """
        if len(ts_data) < 2:
            return None

        model = Prophet(yearly_seasonality=False,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05,
                        interval_width=0.80)

        model.fit(ts_data)

        future = model.make_future_dataframe(periods=periods, freq='YS')
        forecast = model.predict(future)

        # --- FIX 1: Add historical 'y' values to forecast ---
        # This solves the "KeyError: 'y'" in app.py's national forecast plot
        forecast = pd.merge(forecast,
                            ts_data[['ds', 'y']],
                            on='ds',
                            how='left')
        # ----------------------------------------------------

        # Clip values to be between 0% and 100%
        forecast['yhat'] = forecast['yhat'].clip(lower=0, upper=100)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0,
                                                             upper=100)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0,
                                                             upper=100)

        self.models[region_name] = model
        self.forecasts[region_name] = forecast

        return forecast

    def forecast_all_regions(self,
                             df,
                             aggregation_level='national',
                             periods=3):
        """
        Generate forecasts for all regions at specified aggregation level
        """
        ts_data_dict = self.prepare_time_series_data(df, aggregation_level)

        all_forecasts = {}

        for region_name, ts_data in ts_data_dict.items():
            forecast = self.forecast_dropout_risk(ts_data, periods,
                                                  region_name)
            if forecast is not None:
                all_forecasts[region_name] = forecast

        return all_forecasts

    def get_future_predictions(self, forecast_df, years_ahead=3):
        """
        Extract future predictions from forecast

        Parameters:
        - forecast_df: Prophet forecast DataFrame (must include 'y' column from fix 1)
        - years_ahead: Number of years ahead to extract

        Returns:
        - DataFrame with future predictions only
        """
        forecast_df['year'] = forecast_df['ds'].dt.year

        # --- LOGIC FIX: Base 'future' on latest *data* year, not current system year ---
        # This finds the last year where we had actual data ('y' is not null)
        latest_historical_year = forecast_df[
            forecast_df['y'].notna()]['ds'].dt.year.max()

        future_df = forecast_df[forecast_df['year'] >
                                latest_historical_year].copy()
        # -----------------------------------------------------------------------------

        future_df = future_df.head(years_ahead)

        return future_df[['ds', 'year', 'yhat', 'yhat_lower', 'yhat_upper']]

    def classify_forecasted_risk(self, high_risk_percentage):
        """
        Classify forecasted high risk percentage into Low/Medium/High categories
        """
        # These thresholds can be tuned
        if high_risk_percentage > 40:
            return 'High'
        elif high_risk_percentage > 20:
            return 'Medium'
        else:
            return 'Low'

    def generate_state_level_forecast_summary(self, df, periods=3):
        """
        Generate state-level forecast summary with risk classifications

        --- FIX 3: Reverted to unpivoted structure ---
        This version returns a DataFrame with a 'year' column,
        which solves the "KeyError: 'year'" in your running app.py.
        """
        state_forecasts = self.forecast_all_regions(df, 'state', periods)

        summary_data = []

        if not state_forecasts:
            return pd.DataFrame()  # Return empty if no forecasts

        for state, forecast in state_forecasts.items():
            # Get predictions for years *after* the latest data year
            future_preds = self.get_future_predictions(forecast, periods)

            for _, row in future_preds.iterrows():
                summary_data.append({
                    'state':
                    state,
                    'year':
                    int(row['year']),  # <--- This 'year' column is now present
                    'predicted_high_risk_pct':
                    round(row['yhat'], 2),
                    'lower_bound':
                    round(row['yhat_lower'], 2),
                    'upper_bound':
                    round(row['yhat_upper'], 2),
                    'risk_category':
                    self.classify_forecasted_risk(row['yhat'])
                })

        return pd.DataFrame(summary_data)


def generate_national_forecast(df, periods=3):
    """
    Convenience function to generate national-level forecast
    """
    forecaster = DropoutForecaster()
    ts_data = forecaster.prepare_time_series_data(df, 'national')

    if 'National' in ts_data and len(ts_data['National']) >= 2:
        forecast = forecaster.forecast_dropout_risk(ts_data['National'],
                                                    periods, 'National')
        return forecast

    return None


if __name__ == '__main__':
    # This block will run if the script is executed directly
    try:
        df = pd.read_csv('data/udise_data.csv')

        if 'predicted_risk_label' not in df.columns:
            print(
                "Warning: 'predicted_risk_label' not found. Creating dummy 'dropout_risk' data for forecasting test."
            )
            df['dropout_risk'] = np.random.choice(['Low', 'Medium', 'High'],
                                                  size=len(df),
                                                  p=[0.6, 0.3, 0.1])

        print("Generating dropout risk forecasts...")

        forecaster = DropoutForecaster()

        print("\n1. National-level forecast:")
        national_forecast = generate_national_forecast(df, periods=3)
        if national_forecast is not None:
            future_preds = forecaster.get_future_predictions(
                national_forecast, 3)
            print("Future Predictions (National):")
            print(future_preds[['year', 'yhat', 'yhat_lower', 'yhat_upper']])
        else:
            print("Could not generate national forecast (insufficient data).")

        print("\n2. State-level forecast summary (Unpivoted):")
        state_summary = forecaster.generate_state_level_forecast_summary(
            df, periods=3)
        if state_summary is not None and not state_summary.empty:
            print(state_summary.head(15))

            print("\n3. Saving state forecast summary...")
            state_summary.to_csv('data/forecast_summary.csv', index=False)
            print("Forecast summary saved to data/forecast_summary.csv")
        else:
            print(
                "Could not generate state-level summary (insufficient data).")

    except FileNotFoundError:
        print(
            "Error: 'data/udise_data.csv' not found. Run 'generate_sample_data.py' first."
        )
    except Exception as e:
        print(f"An error occurred: {e}")
