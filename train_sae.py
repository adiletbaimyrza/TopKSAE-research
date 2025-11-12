from overcomplete.sae import SAE, TopKSAE
from overcomplete.sae.trackers import DeadCodeTracker
from overcomplete.sae.train import _compute_reconstruction_error
from overcomplete.metrics import l0_eps
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import WeightedICALossFunction, ZeroLogProbException
from tqdm import tqdm
import time
import os
import torch


def train_sae(
    model: SAE,
    dataset: Dataset,
    loss_fn,
    batch_size=1024,
    lr=5e-4,
    epochs=10,
    clip_grd=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
    )
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(data_loader))

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_wica = 0
        epoch_r2 = 0
        epoch_sparsity = 0

        dead_tracker = DeadCodeTracker(model.nb_concepts, device)

        start_time = time.time()
        for batch in tqdm(data_loader):
            x = batch[0].to(device)
            optimizer.zero_grad()

            z_pre, z, x_hat = model(x)
            loss, wica = loss_fn(x, x_hat, z_pre, z, model.get_dictionary())

            dead_tracker.update(z)

            loss.backward()

            if clip_grd:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grd)

            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()

            if wica:
                epoch_wica += wica
            elif epoch_wica:
                epoch_wica = None
            epoch_r2 += _compute_reconstruction_error(x, x_hat)
            epoch_sparsity += l0_eps(z, 0).sum().item()

        batch_count = len(data_loader)
        avg_loss = epoch_loss / batch_count
        avg_r2 = epoch_r2 / batch_count
        avg_sparsity = epoch_sparsity / batch_count
        wica = epoch_wica / batch_count
        dead_ratio = dead_tracker.get_dead_ratio()
        epoch_duration = time.time() - start_time

        print(
            f"Epoch[{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, "
            f"R2: {avg_r2:.4f}, L0: {avg_sparsity:.4f}, "
            f"Dead Features: {dead_ratio*100:.1f}%, "
            f"WICA: {wica:.4f}, "
            f"Time: {epoch_duration:.4f} seconds"
        )


if __name__ == "__main__":
    dataset = TensorDataset(torch.load("dinov3_activations.pth"))
    n_concepts = 50
    k = 5
    loss = WeightedICALossFunction(0.2, 50, cuda=True, z_dim=n_concepts)

    loss_fn = loss.topk_wica
    sae = TopKSAE(768, nb_concepts=n_concepts, top_k=k, device="cuda")

    try:
        train_sae(sae, dataset, loss_fn=loss_fn, epochs=50)
    except ZeroLogProbException as e:
        torch.save([e.means, e.samples], "error.pth")
        exit

    torch.save(sae.state_dict(), f"SAE_d{n_concepts}_k{k}_{loss_fn.__name__}.pth")
