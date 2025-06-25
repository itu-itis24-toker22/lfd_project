import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.ganomaly import Encoder, Decoder, Discriminator
from utils.data_loader import get_data_loader
from utils.ssim import SSIM

def train(data_path, epochs=50, batch_size=32, lr=0.0002, latent_dim=100, lambda_rec=50, lambda_enc=1, lambda_ssim=0.1, target_class='candle'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loader
    train_loader = get_data_loader(data_path, batch_size=batch_size, is_train=True, class_name=target_class)

    # Models
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    optimizer_G = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.L1Loss()
    ssim_loss_func = SSIM()

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        for i, (imgs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            real_imgs = imgs.to(device)

            # Adversarial ground truths
            valid = torch.ones(real_imgs.size(0), 1, 1, 1, device=device, requires_grad=False)
            fake = torch.zeros(real_imgs.size(0), 1, 1, 1, device=device, requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            latent_i, skips = encoder(real_imgs)
            gen_imgs = decoder(latent_i, skips)

            # Loss measures generator's ability to fool the discriminator
            g_loss_adv = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss_rec = reconstruction_loss(gen_imgs, real_imgs)
            g_loss_ssim = 1 - ssim_loss_func(gen_imgs, real_imgs)
            
            # Latent space reconstruction loss
            latent_o, _ = encoder(gen_imgs)
            g_loss_enc = reconstruction_loss(latent_o, latent_i)

            # Total generator loss
            g_loss = g_loss_adv + lambda_rec * g_loss_rec + lambda_enc * g_loss_enc + lambda_ssim * g_loss_ssim

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    print("Training finished.")
    # Save models
    torch.save(encoder.state_dict(), 'encoder_mife.pth')
    torch.save(decoder.state_dict(), 'decoder_mife.pth')
    torch.save(discriminator.state_dict(), 'discriminator_mife.pth')

if __name__ == '__main__':
    # The user needs to download the VisA dataset and provide the path here.
    # Example: 'C:/Users/stoker/Desktop/VisA_dataset'
    # The dataset should be structured with 'train/good' and 'test/good', 'test/bad' folders.
    DATASET_PATH = 'data'
    # To train on all classes, set target_class=None
    # To train on a specific class, set target_class='candle', 'cashew', etc.
        # Experimenting with higher reconstruction loss and no SSIM
    train(DATASET_PATH, epochs=30, target_class='candle', lambda_rec=100, lambda_enc=1, lambda_ssim=0)
