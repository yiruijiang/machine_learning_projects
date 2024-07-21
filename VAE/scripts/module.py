import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from dataset import transform

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# Training loop
def train_one_epoch(model, optimizer, loss_function, device, epoch):

    model.train()

    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):

        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var, 5)
        loss.backward()
        # print(loss)
        train_loss += loss.item()
        optimizer.step()

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset) * 128:.4f}"
    )

    return model


def save_snapshot(model, name):

    torch.save(model, f"{name}.pt")
