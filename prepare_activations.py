import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.imagenette import Imagenette
import torchvision.transforms as transforms
import timm
import os
from tqdm import tqdm
import torch


def prepare_activations(model: nn.Module, dataset: Dataset, save: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = DataLoader(dataset, batch_size=32, num_workers=os.cpu_count())

    tensor_data = torch.tensor([])
    for batch, _ in tqdm(loader):
        with torch.no_grad():
            activations = model.forward_features(batch.to(device))[:, 5:]
            activations = activations.reshape((activations.shape[0] * activations.shape[1], activations.shape[2]))
        tensor_data = torch.cat([tensor_data, activations.cpu()])

    torch.save(tensor_data, save)


if __name__ == "__main__":
    dinov3 = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True)
    dinov3.eval()
    data_config = timm.data.resolve_model_data_config(dinov3)
    dino_transforms = timm.data.create_transform(**data_config, is_training=False)

    dataset = Imagenette(
        "./data", split="train", download=True, transform=dino_transforms
    )

    prepare_activations(dinov3, dataset, "dinov3_activations.pth")
