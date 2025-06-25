import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=100):
        super(Encoder, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        # Downsampling
        self.conv1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(1024)

        # Latent vector
        self.fc = nn.Linear(1024 * 8 * 8, latent_dim)

    def forward(self, x):
        e1 = self.leaky_relu(self.bn1(self.conv1(x)))
        e2 = self.leaky_relu(self.bn2(self.conv2(e1)))
        e3 = self.leaky_relu(self.bn3(self.conv3(e2)))
        e4 = self.leaky_relu(self.bn4(self.conv4(e3)))
        e5 = self.leaky_relu(self.bn5(self.conv5(e4)))
        
        e5_flat = e5.view(e5.size(0), -1)
        z = self.fc(e5_flat)
        
        return z # Baseline: No skip connections

class Decoder(nn.Module):
    def __init__(self, latent_dim=100, out_channels=3):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # Latent to feature map
        self.fc = nn.Linear(latent_dim, 1024 * 8 * 8)

        # Upsampling (channels adjusted for no skip connections)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)

        self.tanh = nn.Tanh()

    def forward(self, z):
        z_reshaped = self.fc(z).view(z.size(0), 1024, 8, 8)

        d1 = self.relu(self.bn1(self.deconv1(z_reshaped)))
        d2 = self.relu(self.bn2(self.deconv2(d1)))
        d3 = self.relu(self.bn3(self.deconv3(d2)))
        d4 = self.relu(self.bn4(self.deconv4(d3)))
        d5 = self.deconv5(d4)
        out = self.tanh(d5)

        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 8, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)
