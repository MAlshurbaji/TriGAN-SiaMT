import torch
import torch.nn as nn

class FC3_Discriminator(nn.Module):
    def __init__(self):
        super(FC3_Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1),  # Adjust dimensions if necessary
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class FC5_Discriminator(nn.Module):
    """
    Fully Convolutional Discriminator for a binary segmentation task.
    Designed for 128x128 input images, it processes two inputs ('map' and 'feature'),
    combines their features, and produces a single probability output indicating 
    whether the input is from a stroke region or healthy tissue.
    
    Args:
        num_classes (int): Number of channels for the 'map' input (set to 1 for binary).
        ndf (int, optional): Base number of filters. Default is 64.
        n_channel (int, optional): Number of channels for the 'feature' input. Default is 1.
    """
    def __init__(self, num_classes=1, ndf=64, n_channel=1):
        super(FC5_Discriminator, self).__init__()
        
        # Initial convolutions for 'map' and 'feature' inputs.
        self.conv0 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_channel, ndf, kernel_size=4, stride=2, padding=1)
        
        # Additional convolutional layers.
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        
        # Use AvgPool2d with kernel size (8,8) to reduce the 8x8 feature map to 1x1.
        self.avgpool = nn.AvgPool2d((8, 8))
        
        # For binary classification, output a single probability.
        self.classifier = nn.Linear(ndf * 8, 1)
        
        # Activation and dropout layers.
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, map, feature=None):
        if feature is None:
            feature = map
        # Process the 'map' and 'feature' inputs separately.
        map_feature = self.conv0(map)
        image_feature = self.conv1(feature)

        # Combine features by element-wise addition.
        x = map_feature + image_feature
        
        # Pass through subsequent convolutional layers.
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        # Apply average pooling to reduce spatial dimensions to 1x1.
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten.
        
        # Final classifier to produce a single logit, followed by a Sigmoid for probability.
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x


# Modified Discriminator Producing Pixel-Wise Confidence Maps
class ConfidenceDiscriminator(nn.Module):
    def __init__(self, in_channels=1, ndf=64):
        super(ConfidenceDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, ndf, kernel_size=4, stride=2, padding=1),  # Change in_channels to 2
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1, kernel_size=3, stride=1, padding=1),  # Pixel-wise confidence map
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)  # Output shape: (batch_size, 1, H, W)
