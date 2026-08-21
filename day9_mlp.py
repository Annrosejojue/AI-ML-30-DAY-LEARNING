import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular
import webbrowser
import os

# 1. Load the Clinical Data
print("Loading Breast Cancer Dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # 0 = Malignant, 1 = Benign

# Split into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train the "Black Box" Model (Random Forest)
print("Training the Black Box (Random Forest)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

accuracy = rf_model.score(X_test, y_test)
print(f"Model Test Accuracy: {accuracy * 100:.2f}%\n")

# 3. Setup the LIME Explainer
print("Initializing LIME Explainer...")
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Malignant (Cancer)', 'Benign (Healthy)'],
    mode='classification'
)

# 4. Interrogate a Specific Patient's Prediction
# Let's pick Patient #5 from our test set
patient_index = 5
patient_data = X_test.iloc[patient_index]
true_label = y_test[patient_index]

print(f"Interrogating the Black Box for Patient #{patient_index}...")
print(f"True Diagnosis: {'Malignant' if true_label == 0 else 'Benign'}")

# We ask LIME to explain the prediction. 
# It needs the patient's data, and the model's prediction function.
explanation = explainer.explain_instance(
    data_row=patient_data, 
    predict_fn=rf_model.predict_proba,
    num_features=5 # Just show me the top 5 reasons
)

# 5. Save and Show the Explanation
html_filename = f"lime_explanation_patient_{patient_index}.html"
explanation.save_to_file(html_filename)

print("\n========================================================")
print(f"SUCCESS! LIME Explanation saved to '{html_filename}'.")
print("Opening the explanation in your web browser...")
print("========================================================")

# Open the HTML file automatically
webbrowser.open('file://' + os.path.realpath(html_filename))