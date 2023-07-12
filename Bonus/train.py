import argparse
from datasets import MyDataset
from torchvision import transforms
import torchvision.models as models
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from nets.lenet import LeNet
from tqdm import tqdm
import wandb
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for optimizer')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--model', type=str, default='resnet18', choices=['lenet', 'resnet18', 'vgg16', 'alexnet', 'squeezenet1_0'], help='Model architecture')
args = parser.parse_args()

# Initialize WandB
run_name = f"{args.model}-epoch{args.epoch}-lr{args.lr}-bs{args.batch_size}-wd{args.weight_decay}"
wandb.init(project="Traffic Sign Recongnition", name=run_name)

# Define data transformation function with common data augmentation techniques
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Adjust image color
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Create datasets and dataloaders
train_dataset = MyDataset(folder_path="Dataset_2/", csv_file="Dataset_2/Train.csv", transform=transform)
test_dataset = MyDataset(folder_path="Dataset_2/", csv_file="Dataset_2/Test.csv", transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


# Define the model
if args.model == 'lenet':
    model = LeNet()
elif args.model == 'resnet18':
    model = models.resnet18(pretrained=True)
elif args.model == 'vgg16':
    model = models.vgg16(pretrained=True)
elif args.model == 'alexnet':
    model = models.alexnet(pretrained=True)
elif args.model == 'squeezenet1_0':
    model = models.squeezenet1_0(pretrained=True)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Set the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_accuracy = 0.0  # Track the best accuracy
best_f1_score = 0.0  # Track the best F1 score
best_model_path = f"Bonus/checkpoints/{args.model}_epoch{args.epoch}_lr{args.lr}_wd{args.weight_decay}.pt"  # Name the best model checkpoint

# Train the model
num_epochs = args.epoch
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

    tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_accuracy:.4f}")

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

    # Calculate test accuracy and F1 score and confusion matrix
    test_accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    cm = confusion_matrix(true_labels, predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}  - Test F1-score (macro): {f1_macro:.4f}")


    # Save the best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_f1_score = f1_macro
        torch.save(model.state_dict(), best_model_path)

    # Log metrics to WandB
    wandb.log({"Epoch": epoch+1, "Training Loss": epoch_loss, "Training Accuracy": epoch_accuracy, "Test Accuracy": test_accuracy, "Test F1-score (macro)": f1_macro})
    wandb.log({"Confusion Matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=true_labels,
        preds=predictions,
        class_names=np.arange(len(test_dataset.classes)),
        title="Confusion Matrix"
    )})

print(f"Best Model Accuracy: {best_accuracy:.4f} - Best Model F1-score (macro): {best_f1_score:.4f}")
