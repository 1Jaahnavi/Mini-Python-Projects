import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from typing import Tuple, Optional
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPricePredictor:
    """A class to fetch stock data, forecast prices using Prophet, and visualize results."""
    
    def __init__(self, ticker: str, start_date: str, end_date: str):
        """Initialize the predictor with stock ticker and date range.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data: Optional[pd.DataFrame] = None
        self.model: Optional[Prophet] = None
        self.forecast: Optional[pd.DataFrame] = None

    def fetch_data(self) -> None:
        """Fetch historical stock data using yfinance and ensure timezone-naive dates."""
        try:
            logger.info(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date)
            if self.data.empty:
                raise ValueError(f"No data found for {self.ticker}")
            self.data = self.data[['Close']].reset_index()
            # Ensure 'Date' column is timezone-naive
            self.data['Date'] = self.data['Date'].dt.tz_localize(None)
            self.data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
            logger.info("Data fetched successfully")
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def train_model(self, periods: int = 30) -> None:
        """Train a Prophet model for forecasting.
        
        Args:
            periods (int): Number of days to forecast into the future.
        """
        if self.data is None:
            raise ValueError("No data available. Run fetch_data() first.")
        
        try:
            logger.info("Training Prophet model")
            self.model = Prophet(daily_seasonality=True)
            self.model.fit(self.data)
            future = self.model.make_future_dataframe(periods=periods)
            self.forecast = self.model.predict(future)
            logger.info("Model trained successfully")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def visualize(self, output_file: str = "forecast.html") -> None:
        """Visualize historical and forecasted prices using Plotly.
        
        Args:
            output_file (str): File to save the Plotly HTML output.
        """
        if self.forecast is None or self.data is None:
            raise ValueError("No forecast available. Run train_model() first.")
        
        try:
            logger.info("Generating visualization")
            fig = go.Figure()
            # Historical data
            fig.add_trace(go.Scatter(
                x=self.data['ds'], y=self.data['y'],
                mode='lines', name='Historical'
            ))
            # Forecasted data
            fig.add_trace(go.Scatter(
                x=self.forecast['ds'], y=self.forecast['yhat'],
                mode='lines', name='Forecast'
            ))
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=self.forecast['ds'], y=self.forecast['yhat_upper'],
                mode='lines', name='Upper CI', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(0,100,255,0.2)'
            ))
            fig.add_trace(go.Scatter(
                x=self.forecast['ds'], y=self.forecast['yhat_lower'],
                mode='lines', name='Lower CI', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(0,100,255,0.2)'
            ))
            fig.update_layout(
                title=f"Stock Price Forecast for {self.ticker}",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_dark"
            )
            fig.write_html(output_file)
            logger.info(f"Visualization saved to {output_file}")
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            raise

def main():
    """Main function to run the stock price predictor."""
    ticker = "TSLA" # MSFT, AAPL, GOOG
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    predictor = StockPricePredictor(ticker, start_date, end_date)
    predictor.fetch_data()
    predictor.train_model(periods=30)
    predictor.visualize()

if __name__ == "__main__":
    main()
