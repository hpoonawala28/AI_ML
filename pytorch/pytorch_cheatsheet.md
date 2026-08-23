# The Complete PyTorch Cheat Sheet
*Based on "PyTorch for Deep Learning & Machine Learning" (Daniel Bourke, freeCodeCamp)*
*Goal: everything you need to go from raw data to a trained, evaluated, saved, deployed neural net.*

---

## 0. How To Use This Sheet

Jump to **Section 16 (End-to-End Flowchart)** for the order of operations. Every other section is a reference for the matching step. This mirrors your ML cheat sheet structure — same philosophy, PyTorch tools.

---

## 1. Setup & Device-Agnostic Code

```python
import torch
import torch.nn as nn
torch.__version__
torch.cuda.is_available()

# ALWAYS write device-agnostic code — first thing in every script
device = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
```
**Rule:** anything that touches a model or tensor for training must be moved to `device`. Mismatched devices (CPU tensor vs GPU model) is the #1 PyTorch error.

---

## 2. Tensors — The Core Data Structure

### Creating tensors
```python
scalar = torch.tensor(7)
vector = torch.tensor([7, 7])
matrix = torch.tensor([[7,8],[9,10]])
tensor = torch.tensor([[[1,2,3],[3,6,9],[2,4,5]]])

torch.zeros(3,4); torch.ones(3,4)
torch.arange(0, 10, 1)
torch.zeros_like(input=tensor)   # same shape as another tensor

torch.rand(3,4)                          # uniform random
torch.randn(3,4)                         # normal/Gaussian random
```

### Tensor attributes (check these when debugging shape/device errors)
```python
tensor.shape          # or tensor.size()
tensor.dtype           # default float32
tensor.device           # cpu / cuda
tensor.ndim
```

### Tensor datatypes
```python
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float32,
                                device=None, requires_grad=False)
float_16_tensor = float_32_tensor.type(torch.float16)   # for speed/memory (mixed precision)
```

### Operations
```python
tensor + 10; tensor * 10; tensor - 10
torch.add(tensor, 10); torch.mul(tensor, 10)

# Matrix multiplication (the #1 source of shape errors — inner dims must match)
torch.matmul(tensor_a, tensor_b)     # or tensor_a @ tensor_b
tensor_b.T                            # transpose to fix shape mismatches

tensor.sum(); tensor.mean(dtype=torch.float32); tensor.min(); tensor.max()
tensor.argmin(); tensor.argmax()      # index of min/max
```

### Reshaping / manipulating shape
```python
x = torch.arange(1., 10.)
x.reshape(1, 9)                       # reshape (must preserve total elements)
x.view(1, 9)                          # like reshape, but shares memory
torch.stack([x, x, x], dim=0)         # stack tensors along new dim
x.squeeze()                           # remove all dims of size 1
x.unsqueeze(dim=0)                    # add a dim of size 1
x.permute(2, 0, 1)                    # rearrange dims (e.g. HWC -> CHW for images)
```

### Indexing
```python
x = torch.arange(1,10).reshape(1,3,3)
x[0]; x[0][0]; x[0][0][0]
x[:, 0]; x[:, :, 1]; x[:, 1, 1]
```

### NumPy interop
```python
import numpy as np
tensor = torch.from_numpy(np.array([1,2,3]))    # numpy -> tensor (default float64!)
tensor.numpy()                                    # tensor -> numpy
```

### GPU / device movement
```python
tensor_on_gpu = tensor.to(device)
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()  # must move to CPU before .numpy()
```

---

## 3. Autograd — How PyTorch Learns

```python
x = torch.tensor(2.0, requires_grad=True)   # track gradients for this tensor
y = x ** 2
y.backward()          # compute dy/dx via backpropagation
x.grad                # -> 4.0

# In training loops:
optimizer.zero_grad()   # reset gradients (they accumulate by default!)
loss.backward()         # backprop
optimizer.step()        # update weights using gradients

# Turn off gradient tracking for inference (saves memory/compute)
with torch.inference_mode():   # preferred over torch.no_grad() in modern PyTorch
    preds = model(X_test)
```
**Forgetting `optimizer.zero_grad()`** is one of the most common bugs — gradients accumulate across batches otherwise.

---

## 4. The PyTorch Workflow (Bourke's Core Pattern)

This is the pattern that repeats across every section of the course — linear regression, classification, computer vision, custom datasets. Only the libraries/tools used at each step change depending on the problem.

```
1. Get data ready (turn into tensors)
2. Build/pick a model (pick loss fn & optimizer)
3. Fit the model to the data (training loop)
4. Make predictions & evaluate the model
5. Improve through experimentation
6. Save & reload trained model
```

### What library handles each step

| Step | Library / Tool | Notes |
|---|---|---|
| 1. Get data ready | `pandas` (load raw data) → `numpy`/`sklearn.model_selection.train_test_split` → `torch.from_numpy()`/`torch.tensor()` | For images: `torchvision.datasets` + `torchvision.transforms` instead |
| 2. Build model | `torch.nn` (`nn.Module`, `nn.Linear`, `nn.Conv2d`...) | For pretrained models: `torchvision.models` |
| 2. Pick loss fn | `torch.nn` (`nn.MSELoss`, `nn.CrossEntropyLoss`, `nn.BCEWithLogitsLoss`) | See Section 6 |
| 2. Pick optimizer | `torch.optim` (`optim.SGD`, `optim.Adam`) | See Section 7 |
| 3. Fit (train) | Pure Python loop + `torch.autograd` (automatic, runs under the hood) | No special library — it's a `for` loop you write yourself |
| 3. Batching (larger datasets) | `torch.utils.data` (`Dataset`, `DataLoader`) | See Section 9 |
| 4. Predict & evaluate | `torch.inference_mode()` for prediction; `torchmetrics` or `sklearn.metrics` for scoring | `sklearn.metrics` works fine too — just `.numpy()` your tensors first |
| 4. Visualize results | `matplotlib.pyplot` | Standard for loss curves, predictions vs actuals, images |
| 5. Improve (experiment) | Manual tuning, or `torch.utils.tensorboard.SummaryWriter` to track runs | See Section 15 |
| 6. Save/load | `torch.save()` / `torch.load()` + `pathlib.Path` | See Section 13 |

---

## 4b. Worked Example — Linear Regression End-to-End

This walks through all 6 workflow steps on the simplest possible problem (`y = weight * X + bias`), the same way the course introduces the workflow before adding complexity.

### Step 1 — Get data ready
```python
import torch
import numpy as np
from sklearn.model_selection import train_test_split

weight, bias = 0.7, 0.3
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)     # shape: [50, 1]
y = weight * X + bias                               # ground truth relationship

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 2 — Build a model, pick loss fn & optimizer
```python
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_0 = LinearRegressionModel().to(device)

loss_fn = nn.L1Loss()                                        # MAE — good default for regression
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
```

### Step 3 — Fit the model (training loop)
```python
epochs = 200
train_loss_values, test_loss_values, epoch_count = [], [], []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 20 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        print(f"Epoch: {epoch} | Train loss: {loss:.4f} | Test loss: {test_loss:.4f}")
```

### Step 4 — Predict & evaluate
```python
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)

import matplotlib.pyplot as plt

def plot_predictions(train_data=X_train.cpu(), train_labels=y_train.cpu(),
                      test_data=X_test.cpu(), test_labels=y_test.cpu(),
                      predictions=None):
    plt.figure(figsize=(8,5))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Train data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions.cpu(), c="r", s=4, label="Predictions")
    plt.legend()

plot_predictions(predictions=y_preds)

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")   # should converge close together
plt.legend()

# Check learned params vs the true weight/bias used to generate the data
model_0.state_dict()          # should be close to {'weight': 0.7, 'bias': 0.3}
```

### Step 5 — Improve through experimentation
Things to try if `test_loss` isn't converging well: more `epochs`, different `lr`, swap `SGD` for `Adam`, add more layers/hidden units (unnecessary for this toy example, essential for real data).

### Step 6 — Save & reload
```python
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_PATH / "linear_regression_model.pth"
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_0.to(device)

loaded_model_0.eval()
with torch.inference_mode():
    loaded_preds = loaded_model_0(X_test)
assert torch.allclose(y_preds, loaded_preds)   # confirms save/load worked correctly
```

---

## 5. Building Models — `nn.Module`

### Basic pattern every model follows
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.layer_1(x)

model = MyModel().to(device)     # always move model to device
```

### Common layers
```python
nn.Linear(in_features, out_features)          # fully connected layer
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)  # CNN
nn.MaxPool2d(kernel_size=2)
nn.Flatten()
nn.Dropout(p=0.5)                              # regularization
nn.BatchNorm2d(num_features)
nn.Embedding(num_embeddings, embedding_dim)    # for categorical/text -> vectors

# Sequential — quick way to stack layers without writing forward() manually
model = nn.Sequential(
    nn.Linear(10, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1)
).to(device)
```

### Activation functions
```python
nn.ReLU()       # default hidden-layer choice, fast, avoids vanishing gradients
nn.Sigmoid()    # binary classification output (0-1 probability)
nn.Softmax(dim=1)  # multi-class output (probabilities summing to 1)
nn.Tanh()       # -1 to 1, sometimes used in RNNs
nn.LeakyReLU()  # fixes "dying ReLU" problem
```

### Inspecting a model
```python
model.state_dict()          # all learnable parameters (weights & biases)
list(model.parameters())
sum(p.numel() for p in model.parameters())   # total parameter count

# Model summary (needs torchinfo package)
from torchinfo import summary
summary(model, input_size=(32, 3, 224, 224))   # (batch, channels, H, W)
```

---

## 6. Loss Functions

```python
loss_fn = nn.L1Loss()                 # MAE — regression
loss_fn = nn.MSELoss()                # MSE — regression
loss_fn = nn.BCELoss()                # binary classification (needs sigmoid applied first)
loss_fn = nn.BCEWithLogitsLoss()      # binary classification (sigmoid built in — preferred, more stable)
loss_fn = nn.CrossEntropyLoss()       # multi-class classification (softmax built in — don't apply softmax yourself)
```
| Task | Loss Function |
|---|---|
| Regression | `nn.L1Loss()` (MAE) or `nn.MSELoss()` |
| Binary classification | `nn.BCEWithLogitsLoss()` (raw logits in, no manual sigmoid) |
| Multi-class classification | `nn.CrossEntropyLoss()` (raw logits in, no manual softmax) |

---

## 7. Optimizers

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)                   # simple, needs tuning
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)     # momentum helps escape local minima
optimizer = optim.Adam(model.parameters(), lr=0.001)                 # adaptive, usually best default choice
```
**Default recommendation (Bourke's go-to):** start with `Adam(lr=0.001)` — it's forgiving and works well out of the box for most problems.

---

## 8. The Training Loop (memorize this — it's identical everywhere)

```python
epochs = 100

for epoch in range(epochs):
    ### Training
    model.train()
    y_pred = model(X_train)                    # 1. Forward pass
    loss = loss_fn(y_pred, y_train)             # 2. Calculate loss
    optimizer.zero_grad()                       # 3. Zero gradients
    loss.backward()                             # 4. Backpropagation
    optimizer.step()                            # 5. Update weights (gradient descent)

    ### Testing / validation
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f} | Test loss: {test_loss:.4f}")
```
**The 5 training steps in order (never forget the order):**
```
Forward pass -> Calculate loss -> Zero grad -> Backward pass -> Optimizer step
```

### Classification-specific additions (accuracy + converting logits)
```python
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

# Binary: raw model output (logits) -> probabilities -> labels
y_logits = model(X_test)
y_pred_probs = torch.sigmoid(y_logits)
y_preds = torch.round(y_pred_probs)

# Multi-class: logits -> probabilities -> class label
y_logits = model(X_test)
y_pred_probs = torch.softmax(y_logits, dim=1)
y_preds = torch.argmax(y_pred_probs, dim=1)
```

---

## 9. Data — Datasets & DataLoaders

### Turning raw data into tensors, splitting
```python
import torch
from sklearn.model_selection import train_test_split

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Custom Dataset class
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
```

### DataLoader (batches + shuffling)
```python
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for X_batch, y_batch in train_dataloader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    ...
```

### Image folder dataset (for custom image classification, folder-per-class)
```python
from torchvision import datasets, transforms

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),   # data augmentation (train only)
    transforms.ToTensor()                       # converts + scales to [0,1]
])

train_data = datasets.ImageFolder(root="data/train", transform=data_transform)
test_data = datasets.ImageFolder(root="data/test", transform=transforms.Compose([
    transforms.Resize((64,64)), transforms.ToTensor()
]))
class_names = train_data.classes
```

---

## 10. Computer Vision — CNNs

```python
from torchvision import datasets, transforms

# Built-in datasets (great for practice — MNIST, FashionMNIST, CIFAR10)
train_data = datasets.FashionMNIST(root="data", train=True, download=True,
                                    transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True,
                                   transform=transforms.ToTensor())
```

### CNN architecture pattern (TinyVGG-style, from the course)
```python
class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 13 * 13, output_shape)  # shape depends on input size!
        )

    def forward(self, x):
        return self.classifier(self.block_2(self.block_1(x)))
```
**Shape debugging tip:** the flatten -> linear layer shape is the #1 CNN bug. Print `x.shape` after each block during development, or pass a dummy batch through and read the error.

---

## 11. Transfer Learning

```python
import torchvision

# Load a pretrained model + its matching preprocessing transforms
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)
auto_transforms = weights.transforms()      # use the model's expected preprocessing

# Freeze the pretrained "feature extractor" layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace the final classifier head for your own number of classes
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=len(class_names))
).to(device)

# Now only the new classifier head trains — much faster than training from scratch
```
**Why:** pretrained models (ResNet, EfficientNet, ViT) already learned general visual features from ImageNet — you only need to retrain the final layer(s) for your specific classes. Needs far less data and compute than training from scratch.

---

## 12. Evaluation & Metrics

```python
from torchmetrics import Accuracy
import torchmetrics

acc_fn = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)
acc = acc_fn(y_preds, y_true)

# Confusion matrix
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=y_preds, target=y_true)
plot_confusion_matrix(conf_mat=confmat_tensor.numpy(), class_names=class_names)
```
| Task | Metric |
|---|---|
| Regression | MAE (`nn.L1Loss`), MSE, RMSE |
| Binary classification | Accuracy, Precision, Recall, F1, ROC-AUC (`torchmetrics`) |
| Multi-class classification | Accuracy, Confusion Matrix, per-class F1 |

---

## 13. Saving & Loading Models

```python
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_PATH / "model_0.pth"

# Save (recommended: state_dict only, not the whole model object)
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

# Load
loaded_model = MyModel()                       # must recreate the same architecture first
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model.to(device)
```

---

## 14. Going Modular (structuring real projects)

Bourke's course converts notebook code into reusable `.py` scripts once patterns stabilize:
```
going_modular/
├── data_setup.py       # creates DataLoaders
├── model_builder.py    # defines the nn.Module class(es)
├── engine.py           # contains train_step(), test_step(), train() functions
├── utils.py            # save_model() and other helpers
└── train.py            # imports the above, actually runs training
```
```python
# train.py — glue script, run from command line: python train.py
import data_setup, engine, model_builder, utils
import torch

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(...)
model = model_builder.TinyVGG(...).to(device)
engine.train(model=model, train_dataloader=train_dataloader,
             test_dataloader=test_dataloader, optimizer=optimizer,
             loss_fn=loss_fn, epochs=5, device=device)
utils.save_model(model=model, target_dir="models", model_name="model.pth")
```
**Why:** notebooks are great for experimenting; scripts are what you actually ship, version-control, and run repeatably/remotely (GSoC/production code should look like this).

---

## 15. Experiment Tracking & Common Gotchas

### Experiment tracking (TensorBoard)
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/experiment_1")
writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train_loss": train_loss,
                                                       "test_loss": test_loss}, global_step=epoch)
writer.close()
# then in terminal: tensorboard --logdir=runs
```

### Most common PyTorch errors & fixes
| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Matrix multiplication shape mismatch | Check `.shape` of both tensors; use `.T` to transpose |
| `Expected all tensors on same device` | Model on GPU, data on CPU (or vice versa) | `.to(device)` on BOTH model and data |
| Loss is `nan` | Learning rate too high, or unstable loss fn | Lower `lr`, use `BCEWithLogitsLoss` instead of manual sigmoid+BCE |
| Model not learning / loss stuck | Forgot `optimizer.zero_grad()` | Add it before `loss.backward()` |
| Wrong predictions shape | Forgot `model.eval()` + `torch.inference_mode()` during test | Always wrap test-time inference in both |
| Shape error in `nn.Linear` after conv layers | Flatten size doesn't match `in_features` | Print `x.shape` right before the linear layer during debugging |

---

## 16. End-to-End Flowchart — "Just Follow This For Any PyTorch Problem"

```
1.  Setup
    -> import torch, set device = "cuda" if available else "cpu"   (Section 1)

2.  Get data ready
    -> Load raw data -> convert to tensors -> split train/test      (Section 9)
    -> For images: use ImageFolder or torchvision.datasets, apply transforms

3.  Build DataLoaders (batch + shuffle)                              (Section 9)

4.  Pick/build a model
    -> Simple tabular data -> nn.Linear stack                        (Section 5)
    -> Images -> CNN (build your own or use transfer learning)       (Sections 10-11)
    -> model = MyModel().to(device)

5.  Pick loss function & optimizer                                    (Sections 6-7)
    -> Regression -> L1Loss/MSELoss + Adam
    -> Binary classification -> BCEWithLogitsLoss + Adam
    -> Multi-class -> CrossEntropyLoss + Adam

6.  Write the training loop                                           (Section 8)
    -> model.train() -> forward -> loss -> zero_grad -> backward -> step
    -> model.eval() + inference_mode() for validation each epoch

7.  Train for N epochs, watch train vs test loss

8.  Diagnose
    -> Train loss low, test loss high? Overfitting ->
       add Dropout, get more data, data augmentation, simplify model
    -> Both losses high/stuck? Underfitting ->
       bigger model, more epochs, higher lr, check data pipeline for bugs

9.  Evaluate with the right metric                                    (Section 12)
    -> Accuracy / F1 / confusion matrix (classification)
    -> MAE / RMSE (regression)

10. Improve through experimentation
    -> Tune lr, hidden units, epochs, add layers, try transfer learning
    -> Track experiments with TensorBoard if comparing many runs        (Section 15)

11. Save the best model                                                (Section 13)
    -> torch.save(model.state_dict(), path)

12. (For real projects) Go modular                                     (Section 14)
    -> Move notebook code into data_setup.py / model_builder.py / engine.py / train.py

13. Load & predict on new data
    -> Recreate model architecture -> load_state_dict -> .to(device) -> .eval()
    -> Apply the SAME transforms used during training before predicting
```

---

## 17. Quick Reference: Tensor Shape Conventions

| Data Type | Shape Convention |
|---|---|
| Tabular batch | `[batch_size, num_features]` |
| Image batch (PyTorch/CNN) | `[batch_size, channels, height, width]` (NCHW) |
| Grayscale image | `channels = 1`; RGB image | `channels = 3` |
| Text/sequence batch | `[batch_size, seq_len, embedding_dim]` |
| Classification logits output | `[batch_size, num_classes]` |
| Binary classification output | `[batch_size, 1]` (before sigmoid) |

---

*Same philosophy as your ML cheat sheet: only the model architecture, loss function, and data pipeline really change per problem. The training loop skeleton (forward -> loss -> zero_grad -> backward -> step) is identical every single time.*
