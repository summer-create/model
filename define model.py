import torch
import torch.nn as nn
import torch.optim as optim

# U-Net architecture (simplified version for illustration)
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Example layers for U-Net, defining layers for encoding (downsampling) and decoding (upsampling)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Example: RGB image with binary output segmentation
model = UNet(in_channels=3, out_channels=1)  # RGB input, binary mask output

# Example: Grayscale image with multi-class output segmentation
# model = UNet(in_channels=1, out_channels=3)  # Grayscale input, 3 classes output
