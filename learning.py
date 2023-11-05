import torch
from torch import nn
import matplotlib.pyplot as plt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(43)

    # Create *known* parameters
    weight = 0.7
    bias = 0.3

    # Create data
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias
    X[:10], y[:10]

    # Create train/test split
    train_split = int(
        0.8 * len(X)
    )  # 80% of data used for training set, 20% for testing
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    # model
    model0 = LRModel2()

    # Put data and model on target device
    model0.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # loss function
    loss_funciton = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.SGD(model0.parameters(), lr=0.001)

    # tracking experiments for testing
    epoch_count = []
    loss_values = []
    test_loss_values = []

    # training loop
    epochs = 10000
    # 0. Loop over each epoch
    for epoch in range(epochs):
        model0.train()

        # 1. Forward propagation
        y_predictions = model0(X_train)

        # 2. Loss Function
        loss = loss_funciton(y_predictions, y_train)

        # 3. Optimizer
        optimizer.zero_grad()

        # 4. Backward propagation
        loss.backward()

        # 5. Gradient descent
        optimizer.step()

        # Testing
        if epoch % 100 == 0:
            model0.eval()  # set model to evaluation mode
            with torch.inference_mode():
                test_predictions = model0(X_test)
                # 1. Forward propagation
                test_loss = loss_funciton(test_predictions, y_test)
            epoch_count.append(epoch)
            loss_values.append(loss.item())
            test_loss_values.append(test_loss.item())

            # Print loss
            print(
                f"Epoch: {epoch} | Loss: {loss.item():.4f} | Test loss: {test_loss.item():.4f}"
            )

    # Plot the loss curve
    # plot_loss_curve(epoch_count, loss_values, test_loss_values)

    # Plot the final predictions against the data
    print(model0.state_dict())
    with torch.inference_mode():
        predictions = model0(X_test)

    plot_predictions(
        X_train.cpu(),
        y_train.cpu(),
        X_test.cpu(),
        y_test.cpu(),
        predictions=predictions.cpu(),
    )
    save_model(model0, "models/model0.pth")


class LRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


class LRModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(ModelClass, path):
    model = ModelClass()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def plot_predictions(
    train_data,
    train_labels,
    test_data,
    test_labels,
    predictions=None,
):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()


def plot_loss_curve(epoch_count, loss_values, test_loss_values):
    plt.figure(figsize=(10, 7))
    plt.plot(epoch_count, loss_values, label="Train loss", color="red")
    plt.plot(epoch_count, test_loss_values, label="Test loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test loss curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
