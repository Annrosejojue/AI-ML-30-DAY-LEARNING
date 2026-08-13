import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss, confusion_matrix

# 1. Simulate Clinical AI Output (The Final Exam Results)
# Let's say we have 1000 patients (900 healthy, 100 sick)
np.random.seed(42)
y_true_healthy = np.zeros(900)
y_true_sick = np.ones(100)
y_true = np.concatenate([y_true_healthy, y_true_sick])

# Our AI outputs probabilities (0.0 to 1.0)
# Healthy patients generally get low scores, sick patients get high scores
y_prob_healthy = np.random.beta(a=1, b=5, size=900) # Skewed towards 0
y_prob_sick = np.random.beta(a=5, b=2, size=100)    # Skewed towards 1
y_probs = np.concatenate([y_prob_healthy, y_prob_sick])

# 2. Calculate ROC-AUC (The Overall Grade)
auc_score = roc_auc_score(y_true, y_probs)
print(f"ROC-AUC Score: {auc_score:.4f} (1.0 is perfect, 0.5 is random guessing)")

# 3. Calculate Brier Score (The Humility/Calibration Metric)
brier_score = brier_score_loss(y_true, y_probs)
print(f"Brier Score:   {brier_score:.4f} (0.0 is perfect, highly trustworthy)")

# 4. Evaluate a Specific Threshold (e.g., 0.5)
threshold = 0.5
y_pred = (y_probs >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\n--- Clinical Evaluation at 0.5 Threshold ---")
print(f"True Positives (Caught the cancer): {tp}")
print(f"False Negatives (Missed the cancer - FATAL): {fn}")
print(f"False Positives (Scared healthy patients): {fp}")
print(f"True Negatives (Correctly identified healthy): {tn}")
print(f"Sensitivity: {sensitivity*100:.1f}%")
print(f"Specificity: {specificity*100:.1f}%")

# 5. Plot the ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Scaring Healthy People)')
plt.ylabel('True Positive Rate (Catching Sick People)')
plt.title('Receiver Operating Characteristic (ROC) - Clinical Model')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Save the plot instead of just showing it, so you can upload it to GitHub!
plt.savefig('roc_curve_day6.png')
print("\nROC Curve saved as 'roc_curve_day6.png'.")