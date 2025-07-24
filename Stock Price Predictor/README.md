**Stock Price Predictor with Prophet**
This project uses the prophet library to forecast stock prices based on historical data fetched from Yahoo Finance using yfinance. The results are visualized interactively using plotly.
Features

Fetch historical stock data for any ticker symbol.
Forecast future stock prices using Facebook's Prophet model.
Visualize historical and predicted prices with confidence intervals.
Modular design with type hints and logging for robustness.
Handles timezone issues in date columns for compatibility with Prophet.

**Installation**

Clone the repository:
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor


Create and activate a virtual environment:
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate


Install dependencies:
pip install -r requirements.txt



Dependencies
See requirements.txt for the full list. Key libraries:

yfinance: Fetch stock data.
prophet: Time-series forecasting.
plotly: Interactive visualizations.
pandas: Data manipulation.
pytest: Unit testing.

Usage

Run the main script:
python src/stock_predictor.py


The script fetches data for Apple (AAPL) by default, trains a Prophet model, and generates a forecast plot saved as forecast.html.

Customize the ticker, date range, or forecast period by modifying the main() function in src/stock_predictor.py.


Notes

Timezone Handling: The yfinance library may return dates with timezone information, which is not supported by prophet. The script converts the Date column to a timezone-naive format using tz_localize(None) to avoid ValueError: Column ds has timezone specified.

Example Output

The script generates forecast.html, an interactive plot showing historical and predicted stock prices with confidence intervals.

Project Structure

src/stock_predictor.py: Main script for fetching, forecasting, and visualizing.
tests/test_stock_predictor.py: Unit tests.
data/: Directory for cached data (excluded from Git).
requirements.txt: Dependency list.
.gitignore: Excludes unnecessary files.
LICENSE: MIT License.

Testing
Run unit tests to verify functionality:
pytest tests/

Feel free to submit issues or pull requests on GitHub!
