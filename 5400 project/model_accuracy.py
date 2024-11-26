import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def print_dataset_info(data):
    """
    Print comprehensive information about the dataset.
    """
    print("\n--- Dataset Information ---")
    print(f"Total number of rows: {len(data)}")
    print(f"Number of unique companies: {data['company'].nunique()}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    print("\nAvailable columns:")
    for col in data.columns:
        print(f"- {col}")

def feature_engineering(data):
    """
    Perform comprehensive feature engineering.
    """
    # Create a copy to avoid modifying the original dataframe
    df = data.copy()

    # Create the target variable: price direction
    df['price_direction'] = (df.groupby('company')['close_price'].shift(-1) > df['close_price']).astype(int)

    # 1. Momentum Indicators
    df['RSI_14'] = df.groupby('company')['close_price'].transform(lambda x: ta.rsi(x, length=14))
    df['ROC_14'] = df.groupby('company')['close_price'].transform(lambda x: ta.roc(x, length=14))

    # 2. Moving Averages
    for window in [7, 14, 30, 50, 100]:
        df[f'moving_avg_{window}'] = df.groupby('company')['close_price'].transform(lambda x: x.rolling(window=window).mean())
        df[f'ema_{window}'] = df.groupby('company')['close_price'].transform(lambda x: ta.ema(x, length=window))

    # 3. Lagged Sentiment Scores
    for lag in [1, 2, 3]:
        df[f'vader_sentiment_lag{lag}'] = df.groupby('company')['vader_daily_sentiment_score'].shift(lag)
        df[f'finbert_sentiment_lag{lag}'] = df.groupby('company')['finbert_daily_sentiment_score'].shift(lag)

    # 4. Volatility Measures
    df['rolling_std_14'] = df.groupby('company')['close_price'].transform(lambda x: x.rolling(window=14).std())

    # 5. Volume Features (if available)
    if 'volume' in df.columns:
        df['OBV'] = df.groupby('company').apply(lambda x: ta.obv(x['close_price'], x['volume'])).reset_index(level=0, drop=True)
        for lag in [1, 2, 3]:
            df[f'volume_lag{lag}'] = df.groupby('company')['volume'].shift(lag)

    # Drop rows with NaN values
    df = df.dropna()

    return df

def select_features(data):
    """
    Dynamically select features based on dataset columns.
    """
    potential_features = [
        'daily_return', 'volatility', 
        'RSI_14', 'ROC_14',
        'vader_daily_sentiment_score', 'finbert_daily_sentiment_score',
        'vader_sentiment_lag1', 'finbert_sentiment_lag1',
        'vader_sentiment_lag2', 'finbert_sentiment_lag2',
        'vader_sentiment_lag3', 'finbert_sentiment_lag3',
        'rolling_std_14', 'OBV'
    ]

    # Add moving averages and EMA columns
    for window in [7, 14, 30, 50, 100]:
        potential_features.extend([f'moving_avg_{window}', f'ema_{window}'])

    # Add volume features if volume column exists
    if 'volume' in data.columns:
        potential_features.extend(['volume_lag1', 'volume_lag2', 'volume_lag3'])

    # Select only features that exist in the dataset
    features = [col for col in potential_features if col in data.columns]
    
    print("\nSelected Features:")
    for feature in features:
        print(f"- {feature}")
    
    return features

def train_and_evaluate_models(X, y):
    """
    Train and evaluate multiple machine learning models with handling class imbalance.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    # Model training and evaluation
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(random_state=42, scale_pos_weight=len(y_train_smote) / sum(y_train_smote == 1))
    }

    results = {}

    for name, model in models.items():
        # Grid search for hyperparameter tuning
        if name == 'Logistic Regression':
            param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        elif name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        else:  # XGBoost
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 4, 5]
            }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_train_scaled, y_train_smote)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'best_params': grid_search.best_params_
        }
        
        print(f"\n{name} Model Performance:")
        print(f"Accuracy: {results[name]['accuracy']:.2f}")
        print(results[name]['classification_report'])
        print("Best Parameters:", results[name]['best_params'])

    return results

def train_lstm_model(X, y):
    """
    Train and evaluate an LSTM model.
    """
    # Reshape and scale data for LSTM
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Prepare sequences
    X_sequences, y_sequences = [], []
    sequence_length = 10  # Adjust based on your data

    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i+sequence_length])
        y_sequences.append(y.iloc[i+sequence_length])

    X_sequences, y_sequences = np.array(X_sequences), np.array(y_sequences)

    # Split data
    split = int(0.8 * len(X_sequences))
    X_train, X_test = X_sequences[:split], X_sequences[split:]
    y_train, y_test = y_sequences[:split], y_sequences[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    model.fit(X_train, y_train, 
              validation_split=0.2, 
              epochs=50, 
              batch_size=32, 
              callbacks=[early_stopping],
              verbose=0)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nLSTM Model Performance:")
    print(f"Accuracy: {accuracy:.2f}")

    return model

def main():
    # Load data
    data = pd.read_csv('data/feature_engineered_data_updated.csv')

    # Print dataset information
    print_dataset_info(data)

    # Feature engineering
    engineered_data = feature_engineering(data)

    # Select features
    features = select_features(engineered_data)

    # Prepare data for modeling
    X = engineered_data[features]
    y = engineered_data['price_direction']

    # Train and evaluate traditional ML models
    ml_results = train_and_evaluate_models(X, y)

    # Train LSTM model
    try:
        lstm_model = train_lstm_model(X, y)
    except Exception as e:
        print(f"Error training LSTM model: {e}")

if __name__ == "__main__":
    main()



