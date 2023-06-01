import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size_1: int = 3, kernel_size_2: int = 3):
        """
        Initialize the DoubleConv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size_1 (int, Optional): Size of first convolution kernel.
            kernel_size_2 (int, Optional): Size of second convolution kernel.
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size_2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DoubleConv module.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(tensor)


class DownBlock(nn.Module):
    """Encoder block module."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size_1: int = 3, kernel_size_2: int = 3):
        """
        Initialize the Encoder module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DownBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, kernel_size_1, kernel_size_2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, tensor: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass of the Encoder module.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor, torch.Tensor): Tuple containing encoded tensor and skip connection tensor.
        """
        tensor = self.conv(tensor)
        skip = tensor
        tensor = self.pool(tensor)
        return tensor, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size_1: int = 3, kernel_size_2: int = 3):
        """
        Initialize the Decoder module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size_1, kernel_size_2)

    def forward(self, tensor: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Args:
            tensor (torch.Tensor): Input tensor.
            skip (torch.Tensor): Skip connection tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        tensor = self.up(tensor)
        if tensor.shape != skip.shape:
            tensor = tf.resize(tensor, size=skip.shape[2:], antialias=False)
        tensor = torch.cat((skip, tensor), dim=1)
        tensor = self.conv(tensor)
        return tensor


class UNET(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 min_feature: int = 64, num_features: int = 4):
        """
        Initialize the UNET model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            min_feature (int): Minimum number of features.
            num_features (int): Number of scale levels.
        """
        super(UNET, self).__init__()

        features = self._calculate_features(min_feature, num_features)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.bottleneck = DoubleConv(features[-1], 2 * features[-1])
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

        for feature in features:
            self.downs.append(DownBlock(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(UpBlock(2 * feature, feature))

    @staticmethod
    def _calculate_features(min_feature: int = 64, num_features: int = 4) -> list:
        """
        Calculate the number of features at each scale level.

        Args:
            min_feature (int): Minimum number of features.
            num_features (int): Number of scale levels.

        Returns:
            list: List containing the number of features at each scale level.
        """
        return [min_feature * (2 ** idx) for idx in range(num_features)]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNET model.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        skip_connections = []

        for down in self.downs:
            tensor, skip = down(tensor)
            skip_connections.insert(0, skip)

        tensor = self.bottleneck(tensor)

        for up, skip_connection in zip(self.ups, skip_connections):
            tensor = up(tensor, skip_connection)

        return self.final(tensor)

if __name__ == "__main__":
    model = UNET().to('cpu')
    summary(model, (3, 112, 160), device='cpu')