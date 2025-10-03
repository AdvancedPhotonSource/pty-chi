# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self, 
        num_in_channels: int = 1, 
        num_levels: int = 3, 
        base_channels: int = 32, 
        use_batchnorm: bool = False, 
        zero_conv: bool = False,
        sigmoid_on_magnitude: bool = True,
        scaled_tanh_on_phase: bool = True,
        restore_shape: bool = True
    ):
        """
        Convolutional autoencoder model with adjustable number of levels.

        Parameters
        ----------
        num_in_channels : int
            The number of input channels.
        num_levels : int
            The number of levels in the autoencoder.
        base_channels : int
            The base number of channels. In the encoder part, the number of output channels
            of level `i` is `base_channels * 2 ** i`.
        use_batchnorm : bool
            Whether to use batch normalization.
        zero_conv: bool
            If True, a zero-conv layer, i.e., a conv layer with weights and biases initialized
            to zeros, is added at the end of both decoders.
        sigmoid_on_magnitude: bool
            If True, apply a sigmoid function on the magnitude output to confine the values
            between 0 and 1.
        scaled_tanh_on_phase: bool
            If True, apply a tanh function on the phase output and scale it by pi.
        restore_shape: bool
            If True and when the output shape does not match the input, it will be
            scaled to match the input shape.
        """
        super(Autoencoder, self).__init__()
        self.num_levels = num_levels
        self.num_in_channels = num_in_channels
        self.base_channels = base_channels
        self.use_batchnorm = use_batchnorm
        self.zero_conv = zero_conv
        self.sigmoid_on_magnitude = sigmoid_on_magnitude
        self.scaled_tanh_on_phase = scaled_tanh_on_phase
        self.restore_shape = restore_shape

        self.build_encoder()
        self.build_magnitude_decoder()
        self.build_phase_decoder()
        
    def build_encoder(self):
        down_blocks = []
        for level in range(self.num_levels):
            down_blocks += self.get_down_block(level)
        self.encoder = nn.Sequential(
            # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            *down_blocks
        )
        
    def create_generic_decoder_layers(self) -> list[nn.Module]:
        up_blocks = []
        for level in range(self.num_levels - 1, -1, -1):
            up_blocks += self.get_up_block(level)
        decoder_layers = [
            *up_blocks,
            nn.Conv2d(self.base_channels * 2, 1, 3, stride=1, padding=(1, 1)),
        ]
        return decoder_layers
    
    def create_zero_conv_layer(self, num_in_channels, num_out_channels) -> nn.Module:
        zero_conv = nn.Conv2d(num_in_channels, num_out_channels, kernel_size=1)
        torch.nn.init.zeros_(zero_conv.weight)
        torch.nn.init.zeros_(zero_conv.bias)
        return zero_conv
        
    def build_magnitude_decoder(self):
        decoder = self.create_generic_decoder_layers()
        if self.sigmoid_on_magnitude:
            decoder.append(nn.Sigmoid())
        if self.zero_conv:
            decoder.append(self.create_zero_conv_layer(1, 1))
        self.decoder1 = nn.Sequential(*decoder)
        
    def build_phase_decoder(self):
        decoder = self.create_generic_decoder_layers()
        if self.scaled_tanh_on_phase:
            decoder.append(nn.Tanh())
        if self.zero_conv:
            decoder.append(self.create_zero_conv_layer(1, 1))
        self.decoder2 = nn.Sequential(*decoder)

    def get_down_block(self, level):
        """
        Get a list of layers in a downsampling block.

        Parameters
        ----------
        level : int
            0-based level index.
            
        Returns
        -------
        list[nn.Module]
            List of layers in the downsampling block.
        """
        num_in_channels = int(self.base_channels * 2 ** (level - 1))
        num_out_channels = int(self.base_channels * 2 ** level)
        if level == 0:
            num_in_channels = self.num_in_channels

        blocks = []
        blocks.append(nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels,
                      kernel_size=3, stride=1, padding=(1, 1)))
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv2d(num_out_channels, num_out_channels, 3, stride=1, padding=(1, 1)))
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        blocks.append(nn.MaxPool2d((2, 2)))

        return blocks

    def get_up_block(self, level):
        """
        Get a list of layers in a upsampling block.
        Parameters
        ----------
        level : int
            0-based level index.
            
        Returns
        -------
        list[nn.Module]
            List of layers in the upsampling block.
        """
        if level == self.num_levels - 1:
            num_in_channels = self.base_channels * 2 ** level
            num_out_channels = self.base_channels * 2 ** level
        elif level == 0:
            num_in_channels = self.base_channels * 2 ** (level + 1)
            num_out_channels = self.base_channels * 2 ** (level + 1)
        else:
            num_in_channels = self.base_channels * 2 ** (level + 1)
            num_out_channels = self.base_channels * 2 ** level
        num_in_channels = int(num_in_channels)
        num_out_channels = int(num_out_channels)

        blocks = []
        blocks.append(nn.Conv2d(num_in_channels, num_out_channels, 3, stride=1, padding=(1, 1)))
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv2d(num_out_channels, num_out_channels, 3, stride=1, padding=(1, 1)))
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        blocks.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        return blocks
    
    def restore_shape_if_needed(self, x, target_shape):
        """Restore the shape of the input tensor to the target shape.
        """
        if self.restore_shape and x.shape[-2:] != target_shape:
            return torch.nn.functional.interpolate(x, size=target_shape, mode='bilinear')
        return x

    def forward(self, x):
        input_lateral_shape = x.shape[-2:]
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)
        
        amp = self.restore_shape_if_needed(amp, input_lateral_shape)
        ph = self.restore_shape_if_needed(ph, input_lateral_shape)

        if self.scaled_tanh_on_phase:
            ph = ph * torch.pi  # Using tanh activation (-1 to 1) for phase so multiply by pi
        return amp, ph
