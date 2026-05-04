# =============================================================================
# DSA 210 - Phase 4: Machine Learning - Step 2: Logistic Regression (Baseline)
# Project: Evaluating the Impact of Startup Fundamentals and Investor Prestige
#          on Exit Success
# Author: Arad Zeinalifarid
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 2: LOGISTIC REGRESSION (BASELINE MODEL)")
print("=" * 60)

# =============================================================================
# 1. LOAD PREPROCESSED DATA
# =============================================================================
X_train = np.load('X_train.npy')
X_test  = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test  = np.load('y_test.npy')

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print(f"Train size: {X_train.shape} | Test size: {X_test.shape}")

# =============================================================================
# 2. TRAIN LOGISTIC REGRESSION
# =============================================================================
print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    class_weight='balanced',   # handles 5.8:1 imbalance
    max_iter=1000,
    random_state=42,
    C=0.1                      # regularization
)
lr.fit(X_train, y_train)
print("Training complete.")

# =============================================================================
# 3. EVALUATE
# =============================================================================
y_pred      = lr.predict(X_test)
y_pred_prob = lr.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Not Exited', 'Exited']))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# =============================================================================
# 4. CONFUSION MATRIX PLOT
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Exited', 'Exited'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Logistic Regression\nConfusion Matrix', fontsize=12, fontweight='bold')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='#5e81ac', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Logistic Regression\nROC Curve', fontsize=12, fontweight='bold')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('plot_lr_evaluation.png', bbox_inches='tight')
plt.show()
print("Saved: plot_lr_evaluation.png")

# =============================================================================
# 5. FEATURE COEFFICIENTS PLOT
# =============================================================================
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#bf616a' if c > 0 else '#5e81ac' for c in coef_df['coefficient']]
ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Logistic Regression — Feature Coefficients\n(Red = increases exit probability)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('plot_lr_coefficients.png', bbox_inches='tight')
plt.show()
print("Saved: plot_lr_coefficients.png")

# =============================================================================
# 6. SAVE MODEL
# =============================================================================
with open('model_lr.pkl', 'wb') as f:
    pickle.dump(lr, f)
print("Saved: model_lr.pkl")

print(f"\n✅ Logistic Regression complete. ROC-AUC: {roc_auc:.4f}")
