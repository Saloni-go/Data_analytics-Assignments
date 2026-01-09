import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from itertools import combinations
import sys
import warnings
warnings.filterwarnings("ignore")

# =============================================================
# PREPROCESSING
# =============================================================
def preprocess_data(df, is_training=True, encoders=None, scaler=None):
    df = df.copy()

    # Fill missing values
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()

    # Encode Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # Encode Ever_Married and Graduated
    if "Ever_Married" in df.columns:
        df["Ever_Married"] = df["Ever_Married"].map({"Yes": 1, "No": 0})
    if "Graduated" in df.columns:
        df["Graduated"] = df["Graduated"].map({"Yes": 1, "No": 0})

    # Label encoding
    if is_training:
        encoders = {}
        for col in ["Profession", "Spending_Score", "Var_1"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        for col in ["Profession", "Spending_Score", "Var_1"]:
            if col in df.columns and col in encoders:
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
                df[col] = encoders[col].transform(df[col])

    if "ID" in df.columns:
        df = df.drop("ID", axis=1)

    if "Segmentation" in df.columns:
        X = df.drop("Segmentation", axis=1)
        y = df["Segmentation"]
    else:
        X = df
        y = None

    if is_training:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, encoders, scaler


# =============================================================
# ONE-VS-ONE CLASSIFIER
# =============================================================
class OneVsOneClassifier:
    def __init__(self, C=50, gamma=0.01):
        self.C = C
        self.gamma = gamma
        self.classifiers = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        print(f"Training {len(list(combinations(self.classes_, 2)))} classifiers...")

        for class1, class2 in combinations(self.classes_, 2):
            mask = (y == class1) | (y == class2)
            X_pair = X[mask]
            y_pair = y[mask]
            clf = SVC(kernel="rbf", C=self.C, gamma=self.gamma, random_state=42)
            clf.fit(X_pair, y_pair)
            self.classifiers[(class1, class2)] = clf
        return self

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.classes_)))
        class_idx = {cls: i for i, cls in enumerate(self.classes_)}
        for (c1, c2), clf in self.classifiers.items():
            preds = clf.predict(X)
            for i, p in enumerate(preds):
                votes[i, class_idx[p]] += 1
        return self.classes_[np.argmax(votes, axis=1)]


# =============================================================
# PARAMETER TESTING
# =============================================================
def test_parameters(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    best_acc = 0
    best_params = {"C": None, "gamma": None}

    configs = [
        {"C": 10, "gamma": 0.001},
        {"C": 50, "gamma": 0.001},
        {"C": 100, "gamma": 0.001},
        {"C": 10, "gamma": 0.01},
        {"C": 50, "gamma": 0.01},
        {"C": 100, "gamma": 0.01},
    ]

    for cfg in configs:
        ovo = OneVsOneClassifier(C=cfg["C"], gamma=cfg["gamma"])
        ovo.fit(X_tr, y_tr)
        preds = ovo.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"C={cfg['C']}, gamma={cfg['gamma']} → val_acc={acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_params = cfg

    print(f"\nBest Validation Accuracy: {best_acc*100:.2f}%  (C={best_params['C']}, γ={best_params['gamma']})")
    return best_params


# =============================================================
# MAIN
# =============================================================
def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION - MULTICLASS CLASSIFICATION (ONE-VS-ONE SVM)")
    print("=" * 60)

    df = pd.read_csv("Customer_train.csv")
    print(f"\nDataset: {df.shape}")
    print(f"Classes: {df['Segmentation'].value_counts().to_dict()}")

    X, y, encoders, scaler = preprocess_data(df, is_training=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    best = test_parameters(X_train, y_train)

    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)

    ovo = OneVsOneClassifier(C=best["C"], gamma=best["gamma"])
    ovo.fit(X_train, y_train)
    ovo_pred = ovo.predict(X_test)

    # --- METRICS ---
    ovo_acc = accuracy_score(y_test, ovo_pred)
    prec = precision_score(y_test, ovo_pred, average="macro")
    rec = recall_score(y_test, ovo_pred, average="macro")
    f1 = f1_score(y_test, ovo_pred, average="macro")

    print(f"\n🔹 Test Accuracy: {ovo_acc*100:.2f}%")
    print(f"Precision (macro): {prec*100:.2f}%")
    print(f"Recall (macro): {rec*100:.2f}%")
    print(f"F1-score (macro): {f1*100:.2f}%")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, ovo_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, ovo_pred, labels=np.unique(y_test))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("Confusion Matrix - One-vs-One SVM")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"OVO Accuracy: {ovo_acc:.4f}")

    # --- If test file is passed in CMD ---
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print("\n" + "=" * 60)
        print(f"GENERATING TEST PREDICTIONS FOR: {test_file}")
        print("=" * 60)

        df_test = pd.read_csv(test_file)
        X_test_file, _, _, _ = preprocess_data(df_test, is_training=False, encoders=encoders, scaler=scaler)

        preds = ovo.predict(X_test_file)
        pd.DataFrame({"Predicted": preds}).to_csv("ovo.csv", index=False)

        print("\nPredictions saved to 'ovo.csv' successfully!")


# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    main()