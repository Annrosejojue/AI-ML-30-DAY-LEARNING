LIME: Local Interpretable Model-agnostic Explanations.

Model-Agnostic: It treats the underlying AI like a closed box. It doesn't care if the AI is a Random Forest, a Neural Network, or a Support Vector Machine. It only cares about Inputs and Outputs.

Local Surrogate Model: A simple, easily understandable model (like linear regression) trained to approximate the predictions of the underlying black-box model locally around a specific prediction.

Perturbation: The process of slightly altering a data point (adding noise or changing categorical values) to probe how the black box's output changes.