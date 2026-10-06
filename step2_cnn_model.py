import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A standard ResNet block that prevents information loss in deep layers."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # The magic skip-connection!
        return F.relu(out)

class CubeCNN(nn.Module):
    def __init__(self):
        super(CubeCNN, self).__init__()
        
        # INPUT: (Batch, 36, 3, 3)
        self.conv_in = nn.Conv2d(in_channels=36, out_channels=128, kernel_size=3, padding=1)
        
        # Residual layers allow the network to "understand" 3D wraparound better
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        
        self.conv_out = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Flattened size: 256 channels * 3 * 3 = 2304
        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 36, 3, 3)
        
        x = F.relu(self.conv_in(x))
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = F.relu(self.conv_out(x))
        
        x = x.view(batch_size, -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

if __name__ == "__main__":
    dummy_input = torch.zeros((10, 6, 6, 3, 3))
    model = CubeCNN()
    predictions = model(dummy_input)
    print(f"ResNet Output Shape: {predictions.shape}") # Should be [10, 1]
