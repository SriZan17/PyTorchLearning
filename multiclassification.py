import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper import plot_decision_boundary
from torchmetrics import Accuracy

# Hyperparameters
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create dataset
X, y = make_blobs(
    n_samples=1000,
    centers=NUM_CLASSES,
    n_features=NUM_FEATURES,
    random_state=RANDOM_SEED,
)

# Turn into tensors
X = torch.FloatTensor(X).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# Visualize the data
# plt.figure(figsize=(10, 7))
# plt.scatter(
#    X_train[:, 0].cpu(), X_train[:, 1].cpu(), c=y_train.cpu(), cmap=plt.cm.RdYlBu
# )
# plt.show()


# Create model
class NeuralNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=8):
        super().__init__()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_hidden),
            torch.nn.Softmax(),
            torch.nn.Linear(num_hidden, num_classes),
        )

    def forward(self, x):
        return self.stack(x)


# Initialize model
model = NeuralNet(NUM_FEATURES, NUM_CLASSES).to(DEVICE)

# Loss and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    # Forward pass
    y_pred = model(X_train)

    # Compute Loss
    loss = loss_function(y_pred, y_train.long())

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Print progress
    if (epoch + 1) % 50 == 0:
        print(f"Epoch: {epoch+1}, loss = {loss.item():.4f}")

# Evaluation
with torch.no_grad():
    y_pred = model(X_test)
    correct = sum(y_test.long() == torch.argmax(y_pred, dim=1))
    accuracy = correct.item() / len(y_test)
    print(f"Accuracy: {accuracy:.4f}")

# Accuracy using torchmetrics
# accuracy = Accuracy(num_classes=NUM_CLASSES).to(DEVICE)
# acc = accuracy(y_pred, y_test).compute()
# print(f"Accuracy: {acc:.4f}")

# Visualize the results
plt.figure(figsize=(12, 7))
plt.subplot(1, 2, 1)
plt.title("Training Data")
plot_decision_boundary(model, X_train.cpu(), y_train.cpu())
plt.subplot(1, 2, 2)
plt.title("Test Data")
plot_decision_boundary(model, X_test.cpu(), y_test.cpu())
plt.show()
