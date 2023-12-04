import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from helper import accuracy_fn
from timeit import default_timer as timer

BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup training data
train_data = datasets.FashionMNIST(
    root="data",  # where to download data to?
    train=True,  # get training data
    download=True,  # download data if it doesn't exist on disk
    transform=ToTensor(),  # images come as PIL format, we want tensors
    target_transform=None,  # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)
class_names = train_data.classes  # get class names from training data

# Visualize a sample image
# image, label = train_data[0]
# plt.imshow(
#    image.squeeze(), cmap="gray"
# )  # image shape is [1, 28, 28] (colour channels, height, width)
# plt.title(train_data.classes[label])
# plt.show()

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(
    train_data,  # dataset to turn into iterable
    batch_size=BATCH_SIZE,  # how many samples per batch?
    shuffle=True,  # shuffle data every epoch?
)
test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,  # don't necessarily have to shuffle the testing data
)


# Model
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # neural networks like their inputs in vector form
            nn.Linear(
                in_features=input_shape, out_features=hidden_units
            ),  # in_features = number of features in a data sample (784 px)
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)


# Need to setup model with input parameters
# model_0 = FashionMNISTModelV0(
#    input_shape=784,  # one for every pixel (28x28)
#    hidden_units=10,  # how many units in the hiden layer
#    output_shape=len(class_names),  # one for every class
# )
#
# model_0.to(device)  # send model to CPU
# Create a model with non-linear and linear layers
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # flatten inputs into single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)


torch.manual_seed(42)
model_1 = FashionMNISTModelV1(
    input_shape=784,  # number of input features
    hidden_units=10,
    output_shape=len(class_names),  # number of output classes desired
).to(
    device
)  # send model to GPU if it's available  # check model device

loss_fn = (
    nn.CrossEntropyLoss()
)  # this is also called "criterion"/"cost function" in some places
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

# Import tqdm for progress bar
from tqdm.auto import tqdm


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# Move values to device
torch.manual_seed(42)


def eval_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device,
):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {
        "model_name": model.__class__.__name__,  # only works when model was created with a class
        "model_loss": loss.item(),
        "model_acc": acc,
    }


# Set the number of epochs (we'll keep this small for faster training times)
# epochs = 3


# Create training and testing loop
# for epoch in tqdm(range(epochs)):
#    print(f"\nEpoch: {epoch}\n-------")
#    # Training
#    train_loss = 0
#
#    # Add a loop to loop through training batches
#    for batch, (X, y) in enumerate(train_dataloader):
#        X = X.to(device)  # send data to GPU
#        y = y.to(device)
#        model_1.train()
#        # 1. Forward pass
#        y_pred = model_1(X)
#
#        # 2. Calculate loss (per batch)
#        loss = loss_fn(y_pred, y)
#        train_loss += loss  # accumulatively add up the loss per epoch
#
#        # 3. Optimizer zero grad
#        optimizer.zero_grad()
#
#        # 4. Loss backward
#        loss.backward()
#
#        # 5. Optimizer step
#        optimizer.step()
#
#        # Print out how many samples have been seen
#        if batch % 400 == 0:
#            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
#
#    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
#    train_loss /= len(train_dataloader)
#
#    # Testing
#    # Setup variables for accumulatively adding up loss and accuracy
#    test_loss, test_acc = 0, 0
#    model_1.eval()
#    with torch.inference_mode():
#        for X, y in test_dataloader:
#            X = X.to(device)
#            y = y.to(device)
#            # 1. Forward pass
#            test_pred = model_1(X)
#
#            # 2. Calculate loss (accumatively)
#            test_loss += loss_fn(
#                test_pred, y
#            )  # accumulatively add up the loss per epoch
#
#            # 3. Calculate accuracy (preds need to be same as y_true)
#            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
#
#        # Calculations on test metrics need to happen inside torch.inference_mode()
#        # Divide total test loss by length of test dataloader (per batch)
#        test_loss /= len(test_dataloader)
#
#        # Divide total accuracy by length of test dataloader (per batch)
#        test_acc /= len(test_dataloader)
#
#    # Print out what's happening
#    print(
#        f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n"
#    )
#
## Calculate training time
# train_time_end_on_cpu = timer()
# total_train_time_model_0 = print_train_time(
#    start=train_time_start_on_cpu,
#    end=train_time_end_on_cpu,
#    device=str(next(model_1.parameters()).device),
# )
def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(
            y_true=y, y_pred=y_pred.argmax(dim=1)
        )  # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def test_step(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    accuracy_fn: torch.nn.Module = accuracy_fn,
    device: torch.device = device,
):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(
                y_true=y,
                y_pred=test_pred.argmax(dim=1),  # Go from logits -> pred labels
            )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


torch.manual_seed(42)

# Measure time
train_time_start_on_gpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(
        data_loader=train_dataloader,
        model=model_1,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
    )
    test_step(
        data_loader=test_dataloader,
        model=model_1,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
    )

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(
    start=train_time_start_on_gpu, end=train_time_end_on_gpu, device=device
)
# Calculate model 1 results with device-agnostic code
model_1_results = eval_model(
    model=model_1,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device,
)
print(model_1_results)
