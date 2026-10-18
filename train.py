from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix



# ---------------- Configuration ----------------

DATA_DIR = Path("dataset")
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# ---------------- Image preprocessing ----------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

train_data = datasets.ImageFolder(
    DATA_DIR / "train", transform=train_transform
)
valid_data = datasets.ImageFolder(
    DATA_DIR / "valid", transform=eval_transform
)
test_data = datasets.ImageFolder(
    DATA_DIR / "test", transform=eval_transform
)

print("Classes:", train_data.classes)
print("Training images:", len(train_data))
print("Validation images:", len(valid_data))
print("Test images:", len(test_data))

assert train_data.classes == valid_data.classes == test_data.classes

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
)
valid_loader = DataLoader(
    valid_data, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")
)
test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")
)

# ---------------- CNN architecture ----------------

class WildfireCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = WildfireCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

# ---------------- Training and validation ----------------

best_val_accuracy = 0.0
model_path = Path("wildfire_cnn.pth")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    train_accuracy = train_correct / train_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss / train_total:.4f} | "
        f"Train Acc: {train_accuracy:.2%} | "
        f"Val Loss: {val_loss / val_total:.4f} | "
        f"Val Acc: {val_accuracy:.2%}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": train_data.classes,
            "image_size": IMAGE_SIZE,
        }, model_path)
        print("Saved best model!")

# ---------------- Final test evaluation ----------------

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(1).cpu()

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

print("\nTest results:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(train_data.classes))),
    target_names=train_data.classes,
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(train_data.classes))),
))

print(f"Best validation accuracy: {best_val_accuracy:.2%}")
print(f"Model saved to: {model_path}")
