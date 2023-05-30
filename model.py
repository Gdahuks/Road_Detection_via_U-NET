from typing import Sequence

import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


class DoubleConv(nn.Module):
    """
    A double convolution module consisting of two sequential convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        conv (nn.Sequential): Sequential module containing two convolutional layers.

    """
    def __init__(self,
                 in_channels: int, out_channels: int,
                 *args, **kwargs):
        super(DoubleConv, self).__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the double convolution module.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the double convolution.

        """
        return self.conv(tensor)


class UNET(nn.Module):
    """
    Implementation of the U-Net architecture.

    Args:
        in_channels (int): Number of input channels. Default is 3.
        out_channels (int): Number of output channels. Default is 1.
        features (Sequence[int]): Sequence of integers specifying the number of features
            at each level of the U-Net. Default is (64, 128, 256, 512).
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        ups (nn.ModuleList): Module list containing upsample layers.
        downs (nn.ModuleList): Module list containing downsample layers.
        pool (nn.MaxPool2d): Max pooling layer.
        bottleneck (DoubleConv): Double convolution module at the bottleneck of the U-Net.
        final (nn.Conv2d): Final convolutional layer.

    """
    def __init__(self,
                 in_channels: int = 3, out_channels: int = 1,
                 features: Sequence[int] = (64, 128, 256, 512),
                 *args, **kwargs):
        super(UNET, self).__init__(*args, **kwargs)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(2 * feature, feature,
                                               kernel_size=2, stride=2))
            self.ups.append(DoubleConv(2 * feature, feature))

        self.bottleneck = DoubleConv(features[-1], 2 * features[-1])
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the U-Net architecture.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the U-Net.

        """
        skip_connections = []

        for down in self.downs:
            tensor = down(tensor)
            skip_connections.insert(0, tensor)
            tensor = self.pool(tensor)

        tensor = self.bottleneck(tensor)

        for up_id in range(0, len(self.ups), 2):
            tensor = self.ups[up_id](tensor)
            skip_connection = skip_connections[up_id // 2]

            # Resize tensor to match skip connection size
            if tensor.shape != skip_connection.shape:
                tensor = tf.resize(tensor, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, tensor), dim=1)
            tensor = self.ups[up_id + 1](concat_skip)

        return self.final(tensor)
