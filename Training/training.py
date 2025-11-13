import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    precision_recall_curve
)


try:
    df = pd.read_csv('dataset_path')
except FileNotFoundError:
    print("ERROR: File 'asteroid_orbital_data.csv' cant be found.")
    exit()

features_to_drop = ['id', 'name', 'est_diameter_m', 'epoch_osculation', 'orbit_uncertainty']

X = df.drop(columns=features_to_drop + ['is_hazardous'])
y = df['is_hazardous'].astype(int) 

if X['relative_velocity_kms'].isnull().sum() > 0:
    print(f"filling {X['relative_velocity_kms'].isnull().sum()} missing values in 'relative_velocity_kms' with median")
    X['relative_velocity_kms'] = X['relative_velocity_kms'].fillna(X['relative_velocity_kms'].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


count_safe = len(y_train[y_train == 0])
count_haz = len(y_train[y_train == 1])
scale_weight = count_safe / count_haz

print(f"scale weight {scale_weight:.2f}")

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=scale_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred_default = model.predict(X_test_scaled)
y_probs = model.predict_proba(X_test_scaled)[:, 1] 

print(classification_report(y_test, y_pred_default, target_names=['safe', 'hazardous']))


precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

try:
    desired_recall = 1.0
    optimal_idx = np.where(recalls >= desired_recall)[0][-1] 
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"optimal threshold : {optimal_threshold:.4f}")

    y_pred_adjusted = (y_probs >= optimal_threshold).astype(int)

    print(classification_report(y_test, y_pred_adjusted, target_names=['safe', 'hazardous']))

except IndexError:
    print("ERROR")


plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], "b--", label="precision")
plt.plot(recalls[:-1], "g-", label="Recall")
if 'optimal_threshold' in locals():
    
    optimal_precision = precisions[optimal_idx]
    optimal_recall = recalls[optimal_idx]
    plt.plot(optimal_recall, optimal_precision, 'ro', markersize=8, label=f'optimal recall (Recall {optimal_recall:.2f})')

plt.title("precision curve vs. recall")
plt.xlabel("Recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.grid(True)
plt.show()

if 'y_pred_adjusted' in locals():
    cm = confusion_matrix(y_test, y_pred_adjusted)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('prediction')
    plt.ylabel('actual')
    plt.title('Confusion Matrix (Adjusted Model)')
    plt.xticks([0.5, 1.5], ['safe', 'hazardous'])
    plt.yticks([0.5, 1.5], ['safe', 'hazardous'])
    plt.show()