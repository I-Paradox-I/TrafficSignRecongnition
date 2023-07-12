from datasets import MyDataset
from torchvision import transforms
import torchvision.models as models
import torch
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from nets.lenet import LeNet
from tqdm import tqdm

# Define data transformation function
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Create datasets and dataloaders
train_dataset = MyDataset(folder_path="Dataset_2/", csv_file="Dataset_2/Train.csv", transform=transform)
test_dataset = MyDataset(folder_path="Dataset_2/", csv_file="Dataset_2/Test.csv", transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)


# Define the model
# model = LeNet()
model = models.resnet18(pretrained=True)
# model = models.vgg16(pretrained=True)
# model = models.alexnet(pretrained=True)
# model = models.squeezenet1_0(pretrained=True)


# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to train mode
    running_loss = 0.0
    correct_predictions = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track training loss and accuracy
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

        progress_bar.set_postfix(loss=loss.item())

    # Calculate average training loss and accuracy
    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct_predictions / len(train_dataset)

    tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

# Test the model
model.eval()  # Set the model to evaluation mode
predictions = []
true_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_dataloader, desc="Testing", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Track predictions and true labels
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate test accuracy
test_accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")
