import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from collections import Counter
import config


def train_classifier(registry_path: str = config.EMBEDDINGS_REGISTRY,
                     model_output: str = config.CLASSIFIER_MODEL,
                     metrics_output: str = config.METRICS_LOG):

    """SVM Classifier"""
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    data = np.load(registry_path, allow_pickle=True).item()
    X = data['embeddings']
    y = data['labels']

    print(f"Loaded {len(X)} embeddings with {len(np.unique(y))} unique identities")

    # Filter: keep only classes with 2+ samples for stratified split
    class_counts = Counter(y)
    valid_classes = {cls for cls, count in class_counts.items() if count >= 2}
    mask = np.array([label in valid_classes for label in y])

    X_filtered = X[mask]
    y_filtered = y[mask]

    print(f"Filtered to {len(np.unique(y_filtered))} identities with 2+ samples")

    stratify = y_filtered if len(np.unique(y_filtered)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.25, random_state=42, stratify=stratify
    )

    print("Training SVM classifier...")
    classifier = SVC(kernel='linear', C=1.0, probability=True)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(classifier, model_output)
    np.save(metrics_output, {'y_true': y_test, 'y_pred': y_pred})

    print(f"\nModel saved: {model_output}")
    print(f"Metrics saved: {metrics_output}")
    return True


if __name__ == "__main__":
    train_classifier()