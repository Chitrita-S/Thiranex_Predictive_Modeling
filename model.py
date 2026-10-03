import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

print("=" * 70)
print("PREDICTIVE MODELING USING MACHINE LEARNING")
print("=" * 70)

print("\nLoading dataset...")

data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["Target"] = data.target


print("\nDataset loaded successfully.")

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(df["Target"].value_counts())

print("\nTarget Meaning:")
print("0 =", data.target_names[0])
print("1 =", data.target_names[1])

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("\nNo missing values found.")

else:
    print("\nMissing values detected.")

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

if duplicate_count > 0:

    df = df.drop_duplicates()

    print("Duplicate rows removed.")

else:

    print("No duplicate rows found.")

X = df.drop("Target", axis=1)

y = df["Target"]


print("\n" + "=" * 70)
print("FEATURE AND TARGET INFORMATION")
print("=" * 70)

print("\nNumber of Features:", X.shape[1])

print("Number of Records:", X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 70)
print("TRAINING DECISION TREE")
print("=" * 70)

decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

decision_tree.fit(
    X_train_scaled,
    y_train
)

dt_predictions = decision_tree.predict(X_test_scaled)

dt_probabilities = decision_tree.predict_proba(
    X_test_scaled
)[:, 1]

print("Decision Tree training completed.")

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=10
)

random_forest.fit(
    X_train_scaled,
    y_train
)

rf_predictions = random_forest.predict(X_test_scaled)

rf_probabilities = random_forest.predict_proba(
    X_test_scaled
)[:, 1]

print("Random Forest training completed.")

def evaluate_model(model_name, y_true, predictions, probabilities):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions
    )

    recall = recall_score(
        y_true,
        predictions
    )

    f1 = f1_score(
        y_true,
        predictions
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    print("\n" + "-" * 70)

    print(model_name)

    print("-" * 70)

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print(f"AUC       : {auc:.4f}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc
    }

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

dt_results = evaluate_model(
    "Decision Tree",
    y_test,
    dt_predictions,
    dt_probabilities
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions,
    rf_probabilities
)

results = pd.DataFrame([
    dt_results,
    rf_results
])

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results.to_string(index=False))

results.to_csv(
    "model_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=data.target_names
    )
)

cm = confusion_matrix(
    y_test,
    rf_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.title("Random Forest - Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "01_random_forest_confusion_matrix.png",
    dpi=300
)

plt.show()

cm_dt = confusion_matrix(
    y_test,
    dt_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.title("Decision Tree - Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "02_decision_tree_confusion_matrix.png",
    dpi=300
)

plt.show()

dt_fpr, dt_tpr, _ = roc_curve(
    y_test,
    dt_probabilities
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probabilities
)

dt_auc = roc_auc_score(
    y_test,
    dt_probabilities
)

rf_auc = roc_auc_score(
    y_test,
    rf_probabilities
)


plt.figure(figsize=(8, 6))

plt.plot(
    dt_fpr,
    dt_tpr,
    label=f"Decision Tree (AUC = {dt_auc:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.tight_layout()

plt.savefig(
    "03_roc_curve.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title("Model Accuracy Comparison")

plt.xlabel("Machine Learning Model")

plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "04_model_accuracy_comparison.png",
    dpi=300
)

plt.show()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP 10 IMPORTANT FEATURES")
print("=" * 70)

print(
    feature_importance.head(10).to_string(
        index=False
    )
)

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title(
    "Top 10 Random Forest Feature Importances"
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "05_feature_importance.png",
    dpi=300
)

plt.show()

joblib.dump(
    random_forest,
    "random_forest_model.pkl"
)

joblib.dump(
    scaler,
    "feature_scaler.pkl"
)

print("\nTrained Random Forest model saved as:")
print("random_forest_model.pkl")

print("\nFeature scaler saved as:")
print("feature_scaler.pkl")

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print("\nDataset:")
print("Breast Cancer Wisconsin Diagnostic Dataset")

print("\nModels trained:")
print("1. Decision Tree")
print("2. Random Forest")

print("\nEvaluation metrics:")
print("1. Accuracy")
print("2. Precision")
print("3. Recall")
print("4. F1 Score")
print("5. AUC")

print("\nVisualizations generated:")
print("1. Random Forest Confusion Matrix")
print("2. Decision Tree Confusion Matrix")
print("3. ROC Curve")
print("4. Model Accuracy Comparison")
print("5. Feature Importance")

print("\nOutput files generated:")
print("model_comparison.csv")
print("feature_importance.csv")
print("random_forest_model.pkl")
print("feature_scaler.pkl")

print("\n" + "=" * 70)
print("PREDICTIVE MODELING PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)