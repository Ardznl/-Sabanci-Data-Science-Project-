# =============================================================================
# DSA 210 - Phase 4: Machine Learning - Step 3: Random Forest
# Project: Evaluating the Impact of Startup Fundamentals and Investor Prestige
#          on Exit Success
# Author: Arad Zeinalifarid
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 3: RANDOM FOREST CLASSIFIER")
print("=" * 60)

# =============================================================================
# 1. LOAD PREPROCESSED DATA (unscaled — trees don't need scaling)
# =============================================================================
X_train = pd.read_csv('X_train_raw.csv')
X_test  = pd.read_csv('X_test_raw.csv')
y_train = pd.read_csv('y_train_raw.csv').values.ravel()
y_test  = pd.read_csv('y_test_raw.csv').values.ravel()

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print(f"Train size: {X_train.shape} | Test size: {X_test.shape}")

# =============================================================================
# 2. TRAIN RANDOM FOREST
# =============================================================================
print("\nTraining Random Forest (this may take ~1 minute)...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("Training complete.")

# =============================================================================
# 3. EVALUATE
# =============================================================================
y_pred      = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Not Exited', 'Exited']))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# =============================================================================
# 4. CONFUSION MATRIX + ROC CURVE
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Exited', 'Exited'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Random Forest\nConfusion Matrix', fontsize=12, fontweight='bold')

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='#a3be8c', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Random Forest\nROC Curve', fontsize=12, fontweight='bold')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('plot_rf_evaluation.png', bbox_inches='tight')
plt.show()
print("Saved: plot_rf_evaluation.png")

# =============================================================================
# 5. FEATURE IMPORTANCE PLOT
# =============================================================================
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#a3be8c' if f in ['investor_prestige_score', 'is_top_tier_investor']
          else '#5e81ac' for f in importance_df['feature']]
ax.barh(importance_df['feature'], importance_df['importance'],
        color=colors, edgecolor='white')
ax.set_title('Random Forest — Feature Importances\n(Green = Investor Prestige | Blue = Fundamentals)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plot_rf_importance.png', bbox_inches='tight')
plt.show()
print("Saved: plot_rf_importance.png")

# =============================================================================
# 6. SAVE MODEL
# =============================================================================
with open('model_rf.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("Saved: model_rf.pkl")

print(f"\n✅ Random Forest complete. ROC-AUC: {roc_auc:.4f}")
