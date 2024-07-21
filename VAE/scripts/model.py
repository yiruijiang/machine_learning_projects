import torch
import torch.nn as nn


class LinearBlock(nn.Module):

    def __init__(self, input_dim, output_dim, dropout):

        super(LinearBlock, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

        self.norm = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.linear(x)

        x = self.norm(x)

        x = self.activation(x)

        x = self.dropout(x)

        return x


# VAE Model
class LinearVAE(nn.Module):

    def __init__(self, **kwargs):

        super(LinearVAE, self).__init__()

        encoder_hidden_layers = kwargs.get("encoder_hidden_layers", [])

        decoder_hidden_layers = kwargs.get("decoder_hidden_layers", [])

        assert len(decoder_hidden_layers) > 0

        input_dim = kwargs.get("input_dim", None)

        assert input_dim

        self.input_dim = input_dim

        bottleneck_dim = kwargs.get("bottleneck_dim", None)

        assert bottleneck_dim

        dropout = kwargs.get("dropout", 0.5)

        self.dropout = nn.Dropout(p=dropout)

        encoder_hidden_layers = [input_dim] + encoder_hidden_layers

        self.encoder_linear_blocks = nn.ModuleList(
            [
                LinearBlock(h1, h2, dropout)
                for h1, h2 in zip(encoder_hidden_layers[:-1], encoder_hidden_layers[1:])
            ]
        )

        last_input_dim = encoder_hidden_layers[-1]

        self.linear_mu = nn.Linear(last_input_dim, bottleneck_dim)  # Mean μ
        self.linear_sigma = nn.Linear(last_input_dim, bottleneck_dim)  # Log variance σ

        decoder_hidden_layers = [bottleneck_dim] + decoder_hidden_layers + [input_dim]

        self.decoder_linear_blocks = nn.ModuleList(
            [
                LinearBlock(h1, h2, dropout)
                for h1, h2 in zip(decoder_hidden_layers[:-1], decoder_hidden_layers[1:])
            ]
        )

    def encode(self, x):

        for block in self.encoder_linear_blocks:

            x = block(x)

        return self.linear_mu(x), self.linear_sigma(x)

    def decode(self, z):

        for block in self.decoder_linear_blocks:

            z = block(z)

        z = torch.sigmoid(z) * 2 - 1

        return z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)

        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):

        mu, log_var = self.encode(x.view(-1, self.input_dim))

        z = self.reparameterize(mu, log_var)

        return self.decode(z), mu, log_var
