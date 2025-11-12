import timm
from torchvision.datasets.imagenette import Imagenette
from overcomplete.sae import TopKSAE
from torch.utils.data.dataloader import DataLoader
from overcomplete.metrics import (
    r2_score,
    l0_eps,
    hoyer,
    frechet_distance,
    energy_of_codes,
)
from overcomplete.sae.trackers import DeadCodeTracker
from loss import WeightedICALossFunction
import torch
from tqdm import tqdm

dinov3 = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True).to(
    "cuda"
)
dinov3.eval()
data_config = timm.data.resolve_model_data_config(dinov3)
dino_transforms = timm.data.create_transform(**data_config, is_training=False)

dataset = Imagenette("./data", split="val", download=True, transform=dino_transforms)

sae = TopKSAE(768, 50, 5, device="cuda").to("cuda")
# sae.load_state_dict(torch.load("SAE_mse.pth"))
# sae.load_state_dict(torch.load("SAE_d50_k5_topk_auxillary_loss_with_wica_report.pth"))
# sae.load_state_dict(torch.load("SAE_d50_k5_topk_wica_v1.pth"))
sae.load_state_dict(torch.load("SAE_d50_k5_topk_wica.pth"))
sae.eval()

dataloader = DataLoader(dataset, batch_size=128, num_workers=16)

wica_loss = WeightedICALossFunction(0.2, 50, cuda=True)
dead_tracker = DeadCodeTracker(50, device="cuda")

r2 = 0
sparsity = 0
wica = 0
sum_hoyer = 0
frechet = 0
energy = 0

with torch.no_grad():
    for batch in tqdm(dataloader):
        imgs = batch[0].to("cuda")
        x = dinov3.forward_features(imgs)[:, 5:]
        n, p, d = x.shape
        x = x.reshape((n * p, d))
        z_pre, z, x_hat = sae(x)

        r2 += r2_score(x, x_hat)
        dead_tracker.update(z)
        sparsity += l0_eps(z_pre, 0).sum()
        wica += wica_loss.loss(z_pre)
        try:
            sum_hoyer += hoyer(z_pre).mean()
        except RuntimeError:
            pass
        frechet += frechet_distance(x, x_hat)
        energy += energy_of_codes(z, sae.get_dictionary())

print(f"R2: {r2 / len(dataloader):.4f}")
print(f"Sparsity: {sparsity / len(dataloader):.4f}")
print(f"Dead ratio: {dead_tracker.get_dead_ratio() * 100:.2f}%")
print(f"WICA: {wica / len(dataloader):.4f}")
print(f"Hoyer: {sum_hoyer / len(dataloader):.4f}")
print(f"frechet: {frechet / len(dataloader):.4f}")

energetics = torch.topk(energy / len(dataloader), 10)
print(f"energy: {" | ".join([f"{i}: {v:.4f}" for i, v in zip(energetics.indices, energetics.values)])}")
