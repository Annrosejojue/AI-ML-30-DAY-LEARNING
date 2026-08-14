import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss


class ClinicalDataProcessor:
    def __init__(self, test_size=0.2, batch_size=32):
        self.test_size = test_size
        self.batch_size = batch_size
        self.scaler = StandardScaler()

    def get_dataloaders(self):
        # 1. Load Data
        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=self.test_size, random_state=42
        )

        # 2. Scale Data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 3. Convert to Tensors
        train_ds = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                                 torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
        test_ds = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), 
                                torch.tensor(y_test, dtype=torch.float32).view(-1, 1))

        # 4. Return Loaders
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        return train_loader, test_ds.tensors[0], test_ds.tensors[1]


class BreastCancerNN(nn.Module):
    def __init__(self, input_features=30):
        super(BreastCancerNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ModelTrainer:
    def __init__(self, model, train_loader, learning_rate=0.01):
        self.model = model
        self.train_loader = train_loader
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, epochs=50):
        print(f"Starting training for {epochs} epochs...")
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
        print("Training Complete.")

    def evaluate_clinical_metrics(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).numpy()
            y_true = y_test.numpy()

            auc = roc_auc_score(y_true, predictions)
            brier = brier_score_loss(y_true, predictions)

            print("\n--- Final Clinical Evaluation ---")
            print(f"ROC-AUC Score: {auc:.4f} (Target: >0.85)")
            print(f"Brier Score:   {brier:.4f} (Target: Closer to 0)")


if __name__ == "__main__":
    # 1. Get Data
    processor = ClinicalDataProcessor()
    train_loader, X_test, y_test = processor.get_dataloaders()

    # 2. Build Model
    model = BreastCancerNN()

    # 3. Train and Evaluate
    trainer = ModelTrainer(model, train_loader)
    trainer.train(epochs=30)
    trainer.evaluate_clinical_metrics(X_test, y_test)