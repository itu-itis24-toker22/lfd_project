import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels=3, ngf=64, latent_dim=100):
        super(Encoder, self).__init__()
        self.e_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.e_conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.e_conv3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.e_conv4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.e_conv5 = nn.Sequential(
            nn.Conv2d(ngf * 8, latent_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attn1 = SelfAttention(ngf * 8)
        self.attn2 = SelfAttention(latent_dim)

    def forward(self, x):
        e1 = self.e_conv1(x)
        e2 = self.e_conv2(e1)
        e3 = self.e_conv3(e2)
        e4 = self.e_conv4(e3)
        e4 = self.attn1(e4) # Apply attention
        e5 = self.e_conv5(e4)
        e5 = self.attn2(e5) # Apply attention
        return e5, (e1, e2, e3, e4)

class Decoder(nn.Module):
    def __init__(self, out_channels=3, ngf=64, latent_dim=100):
        super(Decoder, self).__init__()
        self.d_conv5 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        self.d_conv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.d_conv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.d_conv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.d_conv1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.attn1 = SelfAttention(ngf * 8)
        self.attn2 = SelfAttention(ngf * 4)

    def forward(self, z, skips):
        d5 = self.d_conv5(z)
        d5 = self.attn1(d5) # Apply attention
        d4 = self.d_conv4(torch.cat([d5, skips[3]], 1))
        d4 = self.attn2(d4) # Apply attention
        d3 = self.d_conv3(torch.cat([d4, skips[2]], 1))
        d2 = self.d_conv2(torch.cat([d3, skips[1]], 1))
        d1 = self.d_conv1(torch.cat([d2, skips[0]], 1))
        return d1

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (in_channels) x 256 x 256
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, 1, 8, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)
