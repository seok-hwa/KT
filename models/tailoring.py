from torch import nn

__all__ = ['Tailoring']

class Tailoring(nn.Module):
    def __init__(self, channel, latent_channel):
        super(Tailoring, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channel, latent_channel, 1),
            nn.ReLU(),
            nn.Conv2d(latent_channel, latent_channel, 1),
            nn.ReLU(),
            nn.Conv2d(latent_channel, channel, 1),
        )

    def forward(self, x):
        output = self.layers(x)
        return output