import torch
import torch.nn as nn
import torch.nn.functional as F


def default_down_layer(in_ch, out_ch):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def default_up_layer(in_ch, out_ch):
    return nn.Upsample(scale_factor=2, mode='nearest')

class CAE(nn.Module):
    def __init__(self, in_channels=3, 
                 out_channels = 3, 
                 latent_channels=None, 
                 hidden_channels = None,
                 down_layer_gen = None,
                 up_layer_gen = None
                 ):
        super().__init__()

        hidden_channels = hidden_channels or [32, 64]
        latent_channels = latent_channels or hidden_channels[-1]
        down_layer_gen = down_layer_gen or default_down_layer
        up_layer_gen = up_layer_gen or default_up_layer
      
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
       
        encoder_features = [in_channels] +  hidden_channels
        for feature_in, feature_out in zip(encoder_features, encoder_features[1:]):
            self.encoder.append(NConv(feature_in, feature_out, num_layers=2))
            self.encoder.append(down_layer_gen(feature_out, feature_out))
            
        # latent out
        self.encoder.append(nn.Conv2d(hidden_channels[-1], latent_channels, kernel_size=3, padding=1))
        # latent in
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
    def __init__(self, in_channels=3, out_channels = 3, latent_channels=None, hidden_channels = None):
        super().__init__(in_channels=in_channels, 
                         out_channels=out_channels, 
                         latent_channels=latent_channels, 
                         hidden_channels=hidden_channels)
        
        self.encoder.append(KLBlock(latent_channels))

    def forward(self, x, return_latents = False):
        for layer in self.encoder[:-1]:
            x = layer(x)
        x, mu, logvar = self.encoder[-1](x)
        for layer in self.decoder:
            x = layer(x)
        return (x, mu, logvar) if return_latents else x


class NConv(nn.Module):
    """A stack of Conv2D, BatchNorm, and ReLU layers repeated num_layers times."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_layers=1):
        super(NConv, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, 
                                    out_channels, 
                                    kernel_size=kernel_size, 
                                    stride= stride, 
                                    padding = padding, 
                                    bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)
    

class KLBlock(nn.Module):
    def __init__(self, in_channels):
        super(KLBlock, self).__init__()
        self.conv_mu     = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias = False)
        self.conv_logvar = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias = False)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu     = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        z      = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ConvBlockStack(nn.Module):
    def __init__(self, feature_stages, block_a_spec, block_b_spec):
        super().__init__()
        self.block_a = nn.ModuleList()
        self.block_b = nn.ModuleList()
        block_a_fn, block_a_args = block_a_spec
        block_b_fn, block_b_args = block_b_spec

        for (a_in, a_out), (b_in, b_out) in feature_stages:
            self.block_a.append(block_a_fn(a_in, a_out, **block_a_args))
            self.block_b.append(block_b_fn(b_in, b_out, **block_b_args))

    def forward(self, x, skip=None):
        use_skips = isinstance(skip, list)
        return_skips = skip is True
        skips = []

        for a_layer, b_layer in zip(self.block_a, self.block_b):
            x = a_layer(x)

            if use_skips:
                x = torch.cat((skip.pop(), x), dim=1)
            if return_skips:
                skips.append(x)

            x = b_layer(x)

        return (x, skips) if return_skips else x


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims):
        super(Encoder, self).__init__()

        self.encoder = nn.ModuleList()
        for feature_in, feature_out in zip([in_channels] + hidden_dims[:-1], hidden_dims):
            self.encoder.append(NConv(feature_in, feature_out, kernel_size=3, stride=1, padding=1, num_layers = 1))
            self.encoder.append(NConv(feature_out, feature_out, kernel_size=3, stride=2, padding=1, num_layers = 1))

    def forward(self, x, skips=False):
        for layer in self.encoder:
            x = layer(x)
        return x

    
class Decoder(nn.Module):
    def __init__(self, feature_stages, feature_block_spec, scale_block_spec):
        super(Decoder, self).__init__()

        self.decoder = nn.ModuleList()
        for in_feat, out_feat in zip(feature_stages, feature_stages[1:]):
            self.decoder.append(nn.ConvTranspose2d(in_feat, in_feat, kernel_size=2, stride=2))
            self.decoder.append(NConv(in_feat, out_feat, kernel_size=3, stride=1, padding=1, num_layers = 1))

    def forward(self, x, skips=None):
        for layer in self.decoder:
            x = layer(x)
        return x
    

class IdentityWrapper(nn.Module):
    def __init__(self, layer_cls, **kwargs):
        super().__init__()
        self.layer_cls = layer_cls
        self.kwargs = kwargs

    def __call__(self, in_channels, out_channels):
        return self.layer_cls(**self.kwargs)

