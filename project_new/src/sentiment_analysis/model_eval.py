import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, classification_report, roc_curve, auc, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(input_file):
    """
    Load the input data from a JSON file and prepare TF-IDF vectors and labels.

    Args:
        input_file (str): Path to the JSON file.

    Returns:
        X (DataFrame): Feature vectors (TF-IDF).
        y (array): Encoded polarity labels.
        label_encoder (LabelEncoder): Fitted label encoder.
    """
    print("Loading data...")
    news_data = pd.read_json(input_file)
    X = pd.DataFrame(news_data["tfidf_vector"].tolist())
    le = LabelEncoder()
    y = le.fit_transform(news_data["polarity"])
    return X, y, le


def train_and_evaluate_models(X, y):
    """
    Train SVM, Random Forest, and Naive Bayes classifiers and evaluate their performance.

    Args:
        X (DataFrame): Feature vectors.
        y (array): Encoded labels.

    Returns:
        dict: Model performance results.
    """
    print("Splitting data into train, test, and unknown sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_test, X_unknown, y_test, y_unknown = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Define models
    models = {
        "Support Vector Machine": SVC(kernel="linear", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": MultinomialNB()
    }

    results = {}

    # Train and evaluate models
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")
        model.fit(X_train, y_train)

        # Predictions
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        y_unknown_pred = model.predict(X_unknown)
        y_unknown_proba = model.predict_proba(X_unknown)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        results[model_name] = {
            "Test Accuracy": accuracy_score(y_test, y_test_pred),
            "Test Precision": precision_score(y_test, y_test_pred, average="weighted"),
            "Test Recall": recall_score(y_test, y_test_pred, average="weighted"),
            "Test ROC AUC": roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else "N/A",
            "Confusion Matrix": confusion_matrix(y_test, y_test_pred),
        }

    return results


def plot_confusion_matrices(results, label_encoder, output_dir):
    """
    Plot confusion matrices for all models.

    Args:
        results (dict): Model results containing confusion matrices.
        label_encoder (LabelEncoder): Label encoder for class names.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    for model_name, metrics in results.items():
        cm = metrics["Confusion Matrix"]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(f"{output_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
        plt.close()


def save_results(results, output_file):
    """
    Save the model evaluation results to a text file.

    Args:
        results (dict): Model performance results.
        output_file (str): Path to save the results.

    Returns:
        None
    """
    with open(output_file, "w") as f:
        for model_name, metrics in results.items():
            f.write(f"Results for {model_name}:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write("-" * 50 + "\n")


if __name__ == "__main__":
    # File paths
    INPUT_FILE = "data/news_with_tfidf.json"
    CONFUSION_MATRIX_DIR = "data"
    RESULTS_FILE = "data/model_evaluation_results.txt"

    # Load data
    X, y, label_encoder = load_data(INPUT_FILE)

    # Train and evaluate models
    results = train_and_evaluate_models(X, y)

    # Plot confusion matrices
    plot_confusion_matrices(results, label_encoder, CONFUSION_MATRIX_DIR)

    # Save results to a file
    save_results(results, RESULTS_FILE)

    print("Model evaluation complete! Results and confusion matrices have been saved.")
