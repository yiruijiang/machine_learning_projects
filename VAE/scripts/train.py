import torch.optim as optim
import yaml

from model import LinearVAE
from loss import elbo_loss
from module import save_snapshot, train_one_epoch


def load_config():
    config = yaml.safe_load(open("./model_config.yaml", "rb").read())
    return config


def initialize_model(config):

    device = config["model"]["device"]

    model = LinearVAE(**config["model"]).to(device)

    return model


def initialize_optimizer(config, model):
    lr = float(config["optimizer"]["learning_rate"])
    weight_decay = float(config["optimizer"]["weight_decay"])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def train(model, optimizer, device):

    # Run training
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        model = train_one_epoch(model, optimizer, elbo_loss, device, epoch)

    return model


def main():

    config = load_config()
    model = initialize_model(config)
    optimizer = initialize_optimizer(config, model)
    model = train(model, optimizer, config["model"]["device"])
    save_snapshot(model, "temp")


if __name__ == "__main__":

    main()
