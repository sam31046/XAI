import torch
import torchvision.models as models
from torchvision import transforms
from torch import nn
import random
import numpy as np
from matplotlib import pyplot as plt
from utils import split_model, get_sub_model, get_conditional_modules, load_model_data
import sys
import argparse
import os


parser = argparse.ArgumentParser(description="NeurFlow Setups")
parser.add_argument('--label', type=int, required=True, help='Label index for filtering')
parser.add_argument('--model', type=str, required=True, choices=["resnet50", "googlenet", "alexnet", "swin_large", "resnet50_custom", "resnet50_DR_5class"], help='Model to use')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to store the results')
parser.add_argument('--data_dir', type=str, required=True, help='Directory to store the data')
parser.add_argument('--tau', type=int, required=True, help='The tau value')
parser.add_argument('--keep_data', type=bool, default=True, help='Keep the augmented data in the framework after storing')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the forward pass')
parser.add_argument('--batch_size_ig', type=int, default=8, help='Batch size for the integrated gradients')
parser.add_argument('--device', type=str, default="cuda:0", help='Device to use')
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"

# %% Model Selection Based on Arguments
model_name = args.model
if model_name == "resnet50":
    model = models.resnet50(pretrained=True).eval()
    layers = ["fc", "layer4.2", "layer4.1", "layer4.0", "layer3.5", "layer3.4", "layer3.3", "layer3.2", "layer3.1", "layer3.0", "layer2.3"]
    conditional_modules = get_conditional_modules(model_name)
    
elif model_name in ["resnet50_custom", "resnet50_DR_5class"]:
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    
    if model_name == "resnet50_custom":
        model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
        model_path = 'models/resnet50_diabetic_retinopathy.pth'
    elif model_name == "resnet50_DR_5class":
        model.fc = nn.Linear(num_ftrs, 5)  # 0-4 label
        model_path = 'models/resnet50_DR_5class.pth'
        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    layers = ["fc", "layer4.2", "layer4.1", "layer4.0", "layer3.5", "layer3.4", "layer3.3", "layer3.2", "layer3.1", "layer3.0", "layer2.3"]
    conditional_modules = get_conditional_modules(model_name)
    
elif model_name == "googlenet":
    model = models.googlenet(pretrained=True).eval()
    layers = ["fc", "inception5b", "inception5a", "inception4e", "inception4d", "inception4c", "inception4b", "inception4a", "inception3b", "inception3a", "maxpool2"]
    conditional_modules = get_conditional_modules(model_name)
    
elif model_name == "alexnet":
    model = models.alexnet(pretrained=True).eval()
    layers = ["classifier.6", "classifier.5", "classifier.2", "features.12", "features.9", "features.7", "features.5", "features.2"]
    conditional_modules = get_conditional_modules(model_name)
    
elif model_name == "swin_large":
    from swin import SwinforClassification, ClassificationDataset_All
    model = SwinforClassification(model_name_or_path='microsoft/swin-large-patch4-window12-384-in22k', n_label=7)
    checkpoint = torch.load('/red/ruogu.fang/leem.s/NSF-SCH/Fundus/code/log/run_with_imputation_classification/Swin_classification4loss4.41r20.65.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model = model.to(device)
    layers = [
            "swin.embeddings.norm",  # Layer Normalization after Patch Embedding
            "swin.encoder.layers.0", 
            "swin.encoder.layers.1",  
            "swin.encoder.layers.2",  
            "swin.encoder.layers.3",  
            "swin.encoder.layers.0.blocks.0.layernorm_after", # in first stage, layer norm after first block
            "swin.encoder.layers.0.blocks.1.layernorm_after", # in first stage, layer norm after second block
            # ... 
            "swin.layernorm"]
    conditional_modules = {}
else:
    raise ValueError("Model not supported")

# %% Load Data
target_label = args.label  # label you want to filter/get

if model_name == "swin_large":
    from monai.config import print_config
    from monai.data import DataLoader, DistributedSampler
    from monai.transforms import (
        Activations,
        EnsureChannelFirst,
        AsDiscrete,
        Compose,
        LoadImage,
        Resize,
        RandZoom,
        ScaleIntensity)
    
    img_name_1 = '1000014_21015_0_0.png'
    img_path_1 = os.path.join("/red/ruogu.fang/share/UKB/data/Eye/21015_fundus_left_1_good", img_name_1)
    img_name_2 = '6025140_21015_0_0.png'
    img_path_2 = os.path.join("/red/ruogu.fang/share/UKB/data/Eye/21015_fundus_left_1_good", img_name_2)
    X_test = [img_path_1, img_path_2]  # image path list (N,)
    y_test = np.array([[1], [1]])  # use fisrt label  #[[1,0,0,2,0,2], [1, 0, 0, 2, 0, 1]])  # numpy array (N, 6)
    
    image_size = 384
    val_transforms = Compose(
    [
        LoadImage(image_only=True), 
        EnsureChannelFirst(), 
        Resize((image_size, image_size)),
        ScaleIntensity(),
        #ToTensor(), 
    ])
    
    test_ds = ClassificationDataset_All(X_test, y_test, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=4, num_workers=1)
    with torch.no_grad():
        for _, test_data in enumerate(test_loader):
            test_images, test_labels = (
                test_data[0].to(device, non_blocking=True),
                test_data[1].to(device, non_blocking=True))
            
    print(test_images.shape, test_labels.shape)
    for label in test_labels:
        print(label, type(label))
    images_list = [image.unsqueeze(0) for image in test_images]
    labels_list = [torch.tensor([label]) for label in test_labels]

    all_images = torch.cat(images_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
elif model_name in ["resnet50_custom", "resnet50_DR_5class"]:
    from resnet import DiabeticRetinopathyDataset, load_model_data
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(360),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    csv_path = os.path.join(args.data_dir, "trainLabels_cropped.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_images, all_labels = load_model_data(csv_path, args.data_dir, [target_label], model, device, transform=transform)
else:
    all_images, all_labels = load_model_data(args.data_dir, [target_label], model, device)


# %% Framework and Model Split
from NeurFlow import Framework

start = 1
end = len(layers) - 1

FW = Framework(start, end, all_images, target_label, device, granularities=[1, 0.5, 0.25])
for layer in range(end, start-1, -1):
    index = len(layers) - layer
    netA, _ = split_model(model, layers[index], True, conditional_modules)
    netB = get_sub_model(model, layers[index], layers[index-1], True, conditional_modules)
    print("Layer: ", layer, " ", layers[index])
    FW.fit(
        layer, 
        netA, 
        netB, 
        top_images_per_neuron=50, 
        max_top_neurons=args.tau,
        acceptable_error_range=-1,
        batch_size_ig=args.batch_size_ig,
        batch_size=args.batch_size
    )

# %% Save the Results
if not args.keep_data:
    FW.concept_data = None # type: ignore

store = {
    "FW": FW,
    "labels": target_label,
    "layers": layers,
}
os.makedirs(args.output_dir, exist_ok=True)
torch.save(store, f"{args.output_dir}/store_{model_name}_label{str(target_label)}_tau{args.tau}.pth")
