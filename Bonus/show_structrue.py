import torchvision.models as models
import torch
from nets.lenet import LeNet
from nets.my_net import MY_NET
import os


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(current_directory)

model_names = ["alexnet", "lenet", "my_net", "resnet18", "squeezenet1_0", "vgg16", "vit_b_16"]
checkpoint_paths = [
    current_directory+"/checkpoints/alexnet_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/lenet_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/my_net_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/resnet18_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/squeezenet1_0_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/vgg16_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/vit_b_16_epoch10_lr0.0001_wd0.001.pt"
]

model_list = []  # stroring the models

# Set the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the directory to save the model info files
info_directory = os.path.join(current_directory, "nets")

# Load and initialize the models
for model_name, checkpoint_path in zip(model_names, checkpoint_paths):
    # Load the model
    if model_name == 'alexnet':
        model = models.alexnet()
    elif model_name == 'lenet':
        model = LeNet()
    elif model_name == 'my_net':
        model = MY_NET()
    elif model_name == 'resnet18':
        model = models.resnet18()
    elif model_name == 'squeezenet1_0':
        model = models.squeezenet1_0()
    elif model_name == 'vgg16':
        model = models.vgg16()
    elif model_name == 'vit_b_16':
        model = models.vit_b_16()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # Define the model info file path
    info_file_path = os.path.join(info_directory, f"{model_name}_intro.txt")

    # Save the model structure to the info file
    with open(info_file_path, "w") as f:
        f.write(f"Structure of model: {model_name}\n")
        f.write(str(model))

    # Print the model structure
    print(f"Structure of model: {model_name}\n{model}\n")
