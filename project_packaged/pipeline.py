from financial_news_sentiment.data_collection.get_news import fetch_news
from financial_news_sentiment.data_collection.get_stock_price import fetch_stock_prices
from data.preprocess import preprocess_data
from data.sentiment_feature_eng import sentiment_analysis_and_feature_engineering
from financial_news_sentiment.models.stock_model import train_and_evaluate_models

def run_pipeline():
    # Set API Key and Dates for Data Collection
    api_key = '328c27b7590e4abaac27d06a4ae2a8fd'  # Replace with your API key
    start_date = "2024-10-26"
    end_date = "2024-11-23"
    
    # Define paths for raw, processed, feature-engineered, and model output data
    raw_news_path = 'data/raw/Financial_news.csv'
    raw_stock_path = 'data/raw/historical_stock_prices.csv'
    processed_data_path = 'data/processed/Processed_merged_data.csv'
    feature_engineered_data_path = 'data/processed/Feature_engineered_data.csv'
    model_output_path = 'results/models/stock_prediction_best_model.pkl'
    visualization_output_path = 'results/visualizations/'

    # Fetch the news articles
    fetch_news(api_key, start_date, end_date, raw_news_path)

    # Fetch historical stock prices
    fetch_stock_prices("2024-10-26", "2024-11-23", raw_stock_path)

    # Preprocess and merge the fetched data
    preprocess_data(raw_news_path, raw_stock_path, processed_data_path)

    # Perform sentiment analysis and feature engineering
    sentiment_analysis_and_feature_engineering(processed_data_path, feature_engineered_data_path)

    # Train and evaluate models, and create visualizations
    train_and_evaluate_models(feature_engineered_data_path, model_output_path, visualization_output_path)

if __name__ == "__main__":
    run_pipeline()
