# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptychi.reconstructors.nn.components import DoubleConv


class Down(nn.Module):
    """Downscaling with maxpool then double conv.
    
    This implementation is adapted from https://github.com/milesial/Pytorch-UNet.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv.
    
    This implementation is adapted from https://github.com/milesial/Pytorch-UNet.
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bilinear: bool = True, 
        skip_connection: bool = True,
    ):
        super().__init__()
        self.skip_connection = skip_connection

        # if bilinear, use the normal convolutions to reduce the number of channels
        up_out_channels = in_channels // 2 if skip_connection else in_channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, up_out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """The forward pass.

        Parameters
        ----------
        x1 : Tensor
            The output of the previous layer.
        x2 : Tensor
            The output of the layer in the downsampling part that is on the same scale.
            When skip connection is enabled, this tensor is concatenated with the output.
            Even when skip connection is disabled, this argument is still required because
            it is used to determine the padding of the input tensor.

        Returns
        -------
        Tensor
            The output of the layer.
        """
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.skip_connection:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self, 
        num_in_channels: int, 
        num_out_channels: int, 
        bilinear: bool = True, 
        initialize: bool = False,
        skip_connections: bool = True,
    ):
        """U-net model based on 
        Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for 
        Biomedical Image Segmentation. arXiv:1505.04597.
        
        This implementation is adapted from https://github.com/milesial/Pytorch-UNet.

        Parameters
        ----------
        n_channels : int
            Number of channels in the input image.
        bilinear : bool, optional
            If True, use bilinear upsampling. Otherwise, use transposed convolutions.
        initialize : bool, optional
            If True, initialize the model with normal weights.
        skip_connections : bool, optional
            If True, use skip connections.
        """
        super(UNet, self).__init__()
        self.n_in_channels = num_in_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(num_in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        if skip_connections:
            self.up1 = Up(1024, 512 // factor, bilinear, skip_connection=skip_connections)
            self.up2 = Up(512, 256 // factor, bilinear, skip_connection=skip_connections)
            self.up3 = Up(256, 128 // factor, bilinear, skip_connection=skip_connections)
            self.up4 = Up(128, 64, bilinear, skip_connection=skip_connections)
            self.outc = nn.Conv2d(64, num_out_channels, kernel_size=1)
        else:
            self.up1 = Up(1024 // factor, 512 // factor, bilinear, skip_connection=skip_connections)
            self.up2 = Up(512 // factor, 256 // factor, bilinear, skip_connection=skip_connections)
            self.up3 = Up(256 // factor, 128 // factor, bilinear, skip_connection=skip_connections)
            self.up4 = Up(128 // factor, 64, bilinear, skip_connection=skip_connections)
            self.outc = nn.Conv2d(64, num_out_channels, kernel_size=1)
        
        if initialize:
            self.apply(self._initialize_weights)
            
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 1e-2)
            if m.bias is not None:
                nn.init.normal_(m.bias, 0, 1e-2)

    def forward(self, x):
        """Forward pass of the U-net model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_in_channels, height, width).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_out_channels, height, width).
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
