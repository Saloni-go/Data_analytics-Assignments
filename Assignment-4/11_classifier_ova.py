import numpy as np, pandas as pd, sys
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
import warnings; warnings.filterwarnings("ignore")

# ---------- PREPROCESS ----------
def preprocess_data(df, is_training=True, encoders=None, scaler=None):
    df = df.copy()
    df = df.fillna(method="ffill").fillna(method="bfill")

    if "Gender" in df:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    for col in ["Ever_Married", "Graduated"]:
        if col in df:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    if is_training:
        encoders = {}
        for c in ["Profession", "Spending_Score", "Var_1"]:
            if c in df:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
                encoders[c] = le
    else:
        for c in ["Profession", "Spending_Score", "Var_1"]:
            if c in df and c in encoders:
                df[c] = df[c].astype(str)
                df[c] = df[c].apply(
                    lambda x: x if x in encoders[c].classes_ else encoders[c].classes_[0]
                )
                df[c] = encoders[c].transform(df[c])

    if "ID" in df:
        df.drop("ID", axis=1, inplace=True)
    X = df.drop(columns=["Segmentation"], errors="ignore")
    y = df["Segmentation"] if "Segmentation" in df else None

    if is_training:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, y, encoders, scaler


# ---------- ONE-vs-ALL ----------
class OneVsAllClassifier:
    def __init__(self, C=100, gamma=0.01):
        self.C, self.gamma = C, gamma
        self.classifiers = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        print(f"Training {len(self.classes_)} classifiers (One-vs-All)...")
        for c in self.classes_:
            yb = (y == c).astype(int)
            clf = SVC(
                kernel="rbf", C=self.C, gamma=self.gamma, probability=True, random_state=42
            )
            clf.fit(X, yb)
            self.classifiers[c] = clf
        return self

    def predict(self, X):
        scores = np.column_stack(
            [clf.decision_function(X) for clf in self.classifiers.values()]
        )
        return self.classes_[np.argmax(scores, axis=1)]


# ---------- PARAMETER SEARCH ----------
def tune_params(X, y):
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    grid = [(C, g) for C in [50, 100, 200] for g in [0.01, 0.005, 0.001]]
    best = (0, None)
    for C, g in grid:
        clf = OneVsAllClassifier(C, g).fit(Xtr, ytr)
        acc = accuracy_score(yval, clf.predict(Xval))
        print(f"C={C}, gamma={g}, val_acc={acc*100:.2f}%")
        if acc > best[0]:
            best = (acc, (C, g))
    print(
        f"\nBest Validation Accuracy: {best[0]*100:.2f}%  (C={best[1][0]}, γ={best[1][1]})"
    )
    return best[1]


# ---------- MAIN ----------
if __name__ == "__main__":
    # Load and preprocess training data
    df = pd.read_csv("Customer_train.csv")
    print(f"Dataset: {df.shape}")
    print(f"Classes: {df['Segmentation'].value_counts().to_dict()}")

    X, y, enc, sc = preprocess_data(df, True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Hyperparameter tuning
    bestC, bestG = tune_params(Xtr, ytr)

    # Train final model
    final = OneVsAllClassifier(bestC, bestG).fit(Xtr, ytr)
    pred = final.predict(Xte)

    # ---------- METRICS ----------
    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, average="macro")
    rec = recall_score(yte, pred, average="macro")
    f1 = f1_score(yte, pred, average="macro")

    print("\nEvaluation Metrics:")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-score:  {f1*100:.2f}%")

    print("\nDetailed Classification Report:\n", classification_report(yte, pred))

    # Confusion Matrix
    cm = confusion_matrix(yte, pred, labels=np.unique(yte))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                xticklabels=np.unique(yte), yticklabels=np.unique(yte))
    plt.title("Confusion Matrix - One-vs-All SVM")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # ---------- SAVE PREDICTIONS IF TEST FILE PROVIDED ----------
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"\nLoading test file: {test_file}")
        df_test = pd.read_csv(test_file)
        X_test, _, _, _ = preprocess_data(df_test, is_training=False, encoders=enc, scaler=sc)
        preds = final.predict(X_test)
        pd.DataFrame({"predicted": preds}).to_csv("ova.csv", index=False)
        print("Predictions saved to ova.csv")
    else:
        print("\n(To generate predictions: python ova_classifier.py <test_file>)")