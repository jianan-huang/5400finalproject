import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.utils import shuffle




def visualize_confusion_matrix(y_true, y_pred, model_name, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(os.path.join(output_path, f'{model_name}_confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix for {model_name} saved.")

def plot_feature_importance(model, feature_names, model_name, output_path):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        sorted_indices = importance.argsort()

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance)), importance[sorted_indices], align='center')
        plt.yticks(range(len(importance)), [feature_names[i] for i in sorted_indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance for {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{model_name}_feature_importance.png'))
        plt.close()
        print(f"Feature importance for {model_name} saved.")

def plot_roc_curve(y_true, y_pred_proba, model_name, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.savefig(os.path.join(output_path, f'{model_name}_roc_curve.png'))
    plt.close()
    print(f"ROC Curve for {model_name} saved.")

def train_and_evaluate_models(input_csv_path, model_output_path, visualization_output_path):
    # Load the feature-engineered dataset
    print("Loading feature-engineered dataset...")
    df = pd.read_csv(input_csv_path)

    # Prepare the target variable
    df['target'] = df['daily_return'].apply(lambda x: 1 if x > 0 else 0)

    # Define features and target
    features = [
        'sentiment_score', 'lagged_sentiment_score', 'sentiment_moving_avg_3d',
        'price_moving_avg_5d', 'volatility_5d', 'sentiment_change'
    ]
    X = df[features]
    y = df['target']

    # Split the data into training and test sets
    print("Splitting the dataset into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ensure the visualization output directory exists
    os.makedirs(visualization_output_path, exist_ok=True)

    # Define models to train with hyperparameter tuning
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        "SVC": SVC(kernel='rbf', probability=True, random_state=42)
    }

    # Define hyperparameter grids
    param_grids = {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10]
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ['l2']
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        },
        "SVC": {
            "C": [0.1, 1, 10],
            "gamma": ['scale', 'auto']
        }
    }

    best_model = None
    best_accuracy = 0

    # Train and evaluate each model with hyperparameter tuning
    for model_name, model in models.items():
        print(f"Training {model_name} with hyperparameter tuning...")
        param_grid = param_grids[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)

        # Get the best estimator from grid search
        best_estimator = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")

        # Make predictions on the test set
        y_pred = best_estimator.predict(X_test_scaled)
        y_pred_proba = best_estimator.predict_proba(X_test_scaled)[:, 1] if hasattr(best_estimator, "predict_proba") else None

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.2f}")
        print(f"{model_name} Classification Report:\n{classification_report(y_test, y_pred)}")

        # Visualize confusion matrix
        visualize_confusion_matrix(y_test, y_pred, model_name, visualization_output_path)

        # Plot ROC Curve if probability estimates are available
        if y_pred_proba is not None:
            plot_roc_curve(y_test, y_pred_proba, model_name, visualization_output_path)

        # Plot feature importance for models that provide it
        plot_feature_importance(best_estimator, features, model_name, visualization_output_path)

        # Track the best-performing model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_estimator

    # LSTM Model
    print("Training LSTM model...")
    X_lstm = np.array(X_train_scaled).reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = np.array(X_test_scaled).reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    lstm_model.fit(X_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=1)

    # Evaluate LSTM model
    lstm_accuracy = lstm_model.evaluate(X_test_lstm, y_test, verbose=0)[1]
    print(f"LSTM Model Accuracy: {lstm_accuracy:.2f}")

    if lstm_accuracy > best_accuracy:
        best_model = lstm_model
        best_accuracy = lstm_accuracy

    # Save the best model along with the scaler
    if best_model:
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, 'wb') as model_file:
            pickle.dump({'model': best_model, 'scaler': scaler}, model_file)
        print(f"Best model saved to '{model_output_path}'.")

# Example usage
if __name__ == "__main__":
    input_csv_path = 'data/processed/Feature_engineered_data.csv'
    model_output_path = 'results/models/stock_prediction_best_model.pkl'
    visualization_output_path = 'results/visualizations/'

    train_and_evaluate_models(input_csv_path, model_output_path, visualization_output_path)
