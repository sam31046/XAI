import torch
from torchvision import transforms
from transformers import SwinForImageClassification
import numpy as np


# Specialized model with multiple outpus.
class SwinforClassification(torch.nn.Module):
    def __init__(self, model_name_or_path='microsoft/swin-large-patch4-window12-384-in22k', n_label=7):
        super().__init__()
        model = SwinForImageClassification.from_pretrained(model_name_or_path, num_labels=n_label,
                                                           ignore_mismatched_sizes=True)
        self.swin = model.swin
        self.regressor_sex = torch.nn.Linear(1536, 2)
        self.regressor_smoking = torch.nn.Linear(1536, 2)
        self.regressor_sleeplessness = torch.nn.Linear(1536, 3)
        self.regressor_alcohol = torch.nn.Linear(1536, 3)
        self.regressor_depression = torch.nn.Linear(1536, 2)
        self.regressor_economic_status = torch.nn.Linear(1536, 3)

        self.softmax_sex = torch.nn.Softmax()
        self.softmax_smoking = torch.nn.Softmax()
        self.softmax_sleeplessness = torch.nn.Softmax()
        self.softmax_alcohol = torch.nn.Softmax()
        self.softmax_depression = torch.nn.Softmax()
        self.softmax_economic_status = torch.nn.Softmax()

    def forward(self, x):
        y_swin = self.swin(x).pooler_output
        y_sex = self.regressor_sex(y_swin)
        y_smoking = self.regressor_smoking(y_swin)
        y_sleeplessness = self.regressor_sleeplessness(y_swin)
        y_alcohol = self.regressor_alcohol(y_swin)
        y_depression = self.regressor_depression(y_swin)
        y_economic_status = self.regressor_economic_status(y_swin)

        return y_sex, y_smoking, y_sleeplessness, y_alcohol, y_depression, y_economic_status
    
    
class ClassificationDataset_All(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = np.array(labels)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


if __name__ == '__main__':
    """
    Unit Test
    """
    
    import torch
    from transformers import SwinForImageClassification

    # load model
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-large-patch4-window12-384-in22k"
    )

    # random input
    input_image = torch.randn(1, 3, 384, 384)  # image size 384x384

    # get embeddings layer output
    embeddings_output = model.swin.embeddings(input_image)
    print(embeddings_output)

    # get feature map dim
    height, width = embeddings_output[1]

    print(f"embeddings layer output dim: {embeddings_output[0].shape}, {embeddings_output[1]}")

    # split SwinStage
    stage_output = model.swin.encoder.layers[0](embeddings_output[0], input_dimensions=(height, width))

    # get stage_output first tensor
    stage_tensor = stage_output[0]

    print(f"stage_tensor dim: {stage_tensor.shape}")
    print(f"stage_output dim: {stage_output[2]}")