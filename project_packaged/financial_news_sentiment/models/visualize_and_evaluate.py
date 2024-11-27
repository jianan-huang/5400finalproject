import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.utils import shuffle


# Function to split the dataset manually (instead of using `train_test_split`)
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        X, y = shuffle(X, y, random_state=random_state)

    test_len = int(len(X) * test_size)
    X_train, X_test = X.iloc[:-test_len], X.iloc[-test_len:]
    y_train, y_test = y.iloc[:-test_len], y.iloc[-test_len:]
    
    return X_train, X_test, y_train, y_test

# Ensure the visualization output directory exists
def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

# Visualization functions

def plot_sentiment_trends(df, output_path):
    if 'sentiment_score' in df.columns and 'company' in df.columns and 'date' in df.columns:
        ensure_directory_exists(output_path)
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=df, x='date', y='sentiment_score', hue='company')
        plt.title('Sentiment Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'sentiment_trends.png'))
        plt.close()
        print("Sentiment trends visualization saved.")
    else:
        print("Required columns for sentiment trends visualization are missing. Skipping this plot.")


def plot_price_sentiment_correlation(df, output_path):
    if 'sentiment_score' in df.columns and 'close_price' in df.columns and 'company' in df.columns:
        ensure_directory_exists(output_path)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='sentiment_score', y='close_price', hue='company')
        plt.title('Correlation Between Sentiment and Stock Price')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Stock Price')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'price_sentiment_correlation.png'))
        plt.close()
        print("Price and sentiment correlation visualization saved.")
    else:
        print("Required columns for price-sentiment correlation visualization are missing. Skipping this plot.")


def plot_price_sensitivity(df, output_path):
    if 'sentiment_score' in df.columns and 'daily_return' in df.columns and 'company' in df.columns:
        ensure_directory_exists(output_path)
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='sentiment_score', y='daily_return', hue='company')
        plt.title('Price Sensitivity Analysis by Sentiment Score')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Daily Return')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'price_sensitivity_analysis.png'))
        plt.close()
        print("Price sensitivity analysis visualization saved.")
    else:
        print("Required columns for price sensitivity visualization are missing. Skipping this plot.")


def plot_stock_predictions(df, model, scaler, features, output_path):
    if 'company' in df.columns and 'date' in df.columns:
        ensure_directory_exists(output_path)
        plt.figure(figsize=(14, 7))

        # Prepare the data for prediction
        X = df[features]
        X_scaled = scaler.transform(X)
        df['predicted_movement'] = model.predict(X_scaled)

        sns.lineplot(data=df, x='date', y='predicted_movement', hue='company')
        plt.title('Stock Movement Predictions Based on Sentiment Data')
        plt.xlabel('Date')
        plt.ylabel('Predicted Movement (Up/Down)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'stock_predictions.png'))
        plt.close()
        print("Stock movement prediction visualization saved.")
    else:
        print("Required columns for stock predictions visualization are missing. Skipping this plot.")


# def plot_real_vs_predicted_stock_trend(df, y_test, y_pred, output_path):
#     # Ensure the output directory exists
#     os.makedirs(output_path, exist_ok=True)

#     # Create a DataFrame to align actual and predicted data
#     test_dates = df.loc[y_test.index, 'date']  # Get corresponding dates for the test data
#     test_prices = df.loc[y_test.index, 'close_price']  # Get corresponding prices for the test data

#     # Create a new DataFrame for plotting
#     plot_df = pd.DataFrame({
#         'Date': test_dates,
#         'Actual Stock Price': test_prices,
#         'Predicted Movement': y_pred
#     })

#     # Convert 'Predicted Movement' (0 or 1) into a relative continuous value for better visualization
#     plot_df['Predicted Price'] = plot_df['Actual Stock Price'].shift(1) * (1 + 0.01 * plot_df['Predicted Movement'])

#     # Plot the actual vs. predicted trend
#     plt.figure(figsize=(14, 7))
#     plt.plot(plot_df['Date'], plot_df['Actual Stock Price'], label='Actual Stock Price', color='blue', lw=2)
#     plt.plot(plot_df['Date'], plot_df['Predicted Price'], label='Predicted Price (Based on Movement)', color='red', linestyle='--', lw=2)
#     plt.title('Real vs Predicted Stock Trend (Test Data)')
#     plt.xlabel('Date')
#     plt.ylabel('Stock Price')
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, 'real_vs_predicted_stock_trend_test_data.png'))
#     plt.close()
#     print("Real vs Predicted Stock Trend visualization saved.")
# # Evaluation functions

def evaluate_model_accuracy(y_true, y_pred):
    accuracy = (y_true == y_pred).mean()
    mae = mean_absolute_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Log Error (MSLE): {msle:.2f}")
    return accuracy, mae, msle


def robustness_check(model, X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean Cross-Validation Accuracy: {scores.mean():.2f}")
    return scores

# Plot real vs predicted stock trend
def plot_real_vs_predicted_stock_trend(test_dates, test_prices, y_pred, output_path):
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Create a DataFrame to align actual and predicted data
    plot_df = pd.DataFrame({
        'Date': test_dates,
        'Actual Stock Price': test_prices,
        'Predicted Movement': y_pred
    })

    # Sort by 'Date' to ensure the plot is chronological
    plot_df = plot_df.sort_values(by='Date')

    # Convert 'Predicted Movement' (0 or 1) into a relative continuous value for better visualization
    plot_df['Predicted Price'] = plot_df['Actual Stock Price'].shift(1) * (1 + 0.01 * plot_df['Predicted Movement'])

    # Plot the actual vs. predicted trend
    plt.figure(figsize=(14, 7))
    plt.plot(plot_df['Date'], plot_df['Actual Stock Price'], label='Actual Stock Price', color='blue', lw=2)
    plt.plot(plot_df['Date'], plot_df['Predicted Price'], label='Predicted Price (Based on Movement)', color='red', linestyle='--', lw=2)
    plt.title('Real vs Predicted Stock Trend (Test Data)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'real_vs_predicted_stock_trend_test_data.png'))
    plt.close()
    print("Real vs Predicted Stock Trend visualization saved.")

def time_lag_analysis(df, output_path):
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    companies = df['company'].unique()
    lag_correlations = {}

    for company in companies:
        company_data = df[df['company'] == company].copy()

        # Calculate correlation between sentiment scores and future stock prices for different lags
        lags = range(1, 6)  # 1 to 5-day lag
        correlations = []

        for lag in lags:
            company_data['lagged_sentiment'] = company_data['sentiment_score'].shift(lag)
            correlation = company_data[['lagged_sentiment', 'close_price']].corr().iloc[0, 1]
            correlations.append(correlation)

        lag_correlations[company] = correlations

        # Plotting lag correlations
        plt.figure(figsize=(10, 6))
        plt.plot(lags, correlations, marker='o', linestyle='-', label=f'{company} Correlations')
        plt.xlabel('Lag (days)')
        plt.ylabel('Correlation')
        plt.title(f'Sentiment Time Lag Correlation Analysis - {company}')
        plt.xticks(lags)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'sentiment_time_lag_{company}.png'))
        plt.close()
        print(f"Time lag correlation plot for {company} saved.")

    # Summary of lag correlations
    lag_df = pd.DataFrame(lag_correlations, index=lags)
    lag_df.to_csv(os.path.join(output_path, 'time_lag_correlations.csv'))
    print("Time lag correlations saved as CSV.")

def cross_correlation_analysis(df, output_path):
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    companies = df['company'].unique()

    for company in companies:
        company_data = df[df['company'] == company]

        # Prepare the series
        sentiment_series = company_data['sentiment_score']
        price_series = company_data['close_price']

        # Perform cross-correlation
        cross_correlation = np.correlate(sentiment_series - np.mean(sentiment_series), 
                                         price_series - np.mean(price_series), mode='full')

        # Get lags for visualization purposes
        lags = np.arange(-len(price_series) + 1, len(sentiment_series))

        # Plot the cross-correlation
        plt.figure(figsize=(14, 7))
        plt.plot(lags, cross_correlation, color='purple')
        plt.title(f'Cross-Correlation between Sentiment and Stock Price - {company}')
        plt.xlabel('Lag')
        plt.ylabel('Cross-Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'cross_correlation_{company}.png'))
        plt.close()
        print(f"Cross-correlation plot for {company} saved.")


# Visualization and Evaluation Pipeline
def run_visualization_and_evaluation(feature_engineered_data_path, model_output_path, visualization_output_path):
    # Load the feature-engineered dataset
    df = pd.read_csv(feature_engineered_data_path)

    # Load trained model for evaluation
    with open(model_output_path, 'rb') as model_file:
        model_data = pickle.load(model_file)
    model = model_data['model']
    scaler = model_data['scaler']

    # Prepare feature and target sets for evaluation
    features = [
        'sentiment_score', 'lagged_sentiment_score', 'sentiment_moving_avg_3d',
        'price_moving_avg_5d', 'volatility_5d', 'sentiment_change'
    ]
    X = df[features]
    if 'target' in df.columns:
        y = df['target']
    else:
        print("Target column is missing in the dataset. Skipping evaluation.")
        return

    # Split the data into training and test sets
    print("Splitting the dataset into training and test sets for evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model predictions for evaluation
    y_pred = model.predict(X_test_scaled)

    # Plot real vs predicted stock trend (using test data)
    test_dates = df.loc[y_test.index, 'date']  # Get corresponding dates for the test data
    test_prices = df.loc[y_test.index, 'close_price']  # Get corresponding prices for the test data
    plot_real_vs_predicted_stock_trend(test_dates, test_prices, y_pred, visualization_output_path)

    # Run Sentiment Correlation Check - Time Lag Analysis and Cross-Correlation
    time_lag_analysis(df, visualization_output_path)
    cross_correlation_analysis(df, visualization_output_path)

if __name__ == "__main__":
    feature_engineered_data_path = 'data/processed/Feature_engineered_data.csv'
    model_output_path = 'results/models/stock_prediction_best_model.pkl'
    visualization_output_path = 'results/visualizations/'

    run_visualization_and_evaluation(feature_engineered_data_path, model_output_path, visualization_output_path)