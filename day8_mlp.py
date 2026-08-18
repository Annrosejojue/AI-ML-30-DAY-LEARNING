import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

# 1. Load the Clinical Data
print("Loading Breast Cancer Dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # 0 = Malignant, 1 = Benign

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train the "Glass Box" Model
# Instead of a complex Neural Network, we use an Explainable Boosting Machine (EBM)
print("Training the Explainable Boosting Machine (EBM)...")
ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train, y_train)

# Calculate standard accuracy just to make sure it learned
accuracy = ebm.score(X_test, y_test)
print(f"Model Test Accuracy: {accuracy * 100:.2f}%")

# 3. Generate Explanations
print("\nGenerating Explanations...")

# Global Explanation: How does the AI make decisions in general?
# Which features does it care about the most across ALL patients?
ebm_global = ebm.explain_global(name='Overall Clinical Rules')

# Local Explanation: How did the AI make a decision for specific patients?
# Let's look at the first 5 patients in our test set.
ebm_local = ebm.explain_local(X_test[:5], y_test[:5], name='Individual Patient Diagnoses')

print("\n========================================================")
print("SUCCESS! Launching XAI Dashboard.")
print("A new tab should open in your web browser automatically.")
print("If it doesn't, look for the local URL provided below.")
print("Press CTRL+C in this terminal to shut down the server when you are done.")
print("========================================================")

# 4. Launch the Dashboard
# This will start a local web server and show you the exact scorecards the AI is using!
show([ebm_global, ebm_local])