import torch
import torch.nn as nn
import torch.nn.functional as F


def default_down_layer(in_ch, out_ch):
    """Returns a default downsampling layer."""
    return nn.MaxPool2d(kernel_size=2, stride=2)

def default_up_layer(in_ch, out_ch):
    """Returns a default upsampling layer."""
    return nn.Upsample(scale_factor=2, mode='nearest')


class CAE(nn.Module):
    """Convolutional Autoencoder with customizable encoder/decoder layer generators."""
    def __init__(self, in_channels=3, 
                 out_channels=3, 
                 latent_channels=None, 
                 hidden_channels=None,
                 down_layer_gen=None,
                 up_layer_gen=None):
        super().__init__()

        hidden_channels = hidden_channels or [32, 64]
        latent_channels = latent_channels or hidden_channels[-1]
        down_layer_gen = down_layer_gen or default_down_layer
        up_layer_gen = up_layer_gen or default_up_layer
      
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
       
        encoder_features = [in_channels] + hidden_channels
        for feature_in, feature_out in zip(encoder_features, encoder_features[1:]):
            self.encoder.append(NConv(feature_in, feature_out, num_layers=2))
            self.encoder.append(down_layer_gen(feature_out, feature_out))
            
        self.encoder.append(nn.Conv2d(hidden_channels[-1], latent_channels, kernel_size=3, padding=1))
        self.decoder.append(nn.Conv2d(latent_channels, hidden_channels[-1], kernel_size=3, padding=1))

        decoder_features = hidden_channels[::-1]
        for feature_in, feature_out in zip(decoder_features, decoder_features[1:]):
            self.decoder.append(up_layer_gen(feature_in, feature_in))
            self.decoder.append(NConv(feature_in, feature_out, num_layers=2))

        self.decoder.append(up_layer_gen(hidden_channels[0], hidden_channels[0]))
        self.decoder.append(nn.Conv2d(hidden_channels[0], hidden_channels[0], kernel_size=3, stride=1, padding=1))
        self.decoder.append(nn.Conv2d(hidden_channels[0], out_channels, kernel_size=3, stride=1, padding=1))
        self.decoder.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)
        return x


class VAE(CAE):
    """Variational Autoencoder subclass that adds a KL divergence block to the encoder."""
    def __init__(self, in_channels=3, out_channels=3, latent_channels=None, hidden_channels=None):
        super().__init__(in_channels=in_channels, 
                         out_channels=out_channels, 
                         latent_channels=latent_channels, 
                         hidden_channels=hidden_channels)
        
        self.encoder.append(KLBlock(latent_channels))

    def forward(self, x, return_latents=False):
        for layer in self.encoder[:-1]:
            x = layer(x)
        x, mu, logvar = self.encoder[-1](x)
        for layer in self.decoder:
            x = layer(x)
        return (x, mu, logvar) if return_latents else x


class NConv(nn.Module):
    """A stack of Conv2D → BatchNorm → ReLU blocks repeated num_layers times."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_layers=1):
        super(NConv, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, 
                                    out_channels, 
                                    kernel_size=kernel_size, 
                                    stride=stride, 
                                    padding=padding, 
                                    bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)
    

class KLBlock(nn.Module):
    """Latent sampling block that outputs z, mean, and log-variance from input feature maps."""
    def __init__(self, in_channels):
        super(KLBlock, self).__init__()
        self.conv_mu     = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv_logvar = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def reparameterize(self, mu, logvar):
        """Applies the reparameterization trick to sample from a Gaussian distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu     = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        z      = self.reparameterize(mu, logvar)
        return z, mu, logvar


class IdentityWrapper(nn.Module):
    """Wraps a layer constructor to ignore input/output channel args and return a fixed layer."""
    def __init__(self, layer_cls, **kwargs):
        super().__init__()
        self.layer_cls = layer_cls
        self.kwargs = kwargs

    def __call__(self, in_channels, out_channels):
        return self.layer_cls(**self.kwargs)


