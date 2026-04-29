# import os
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

# import matplotlib.pyplot as plt
import argparse


SEED = 42
BATCH_SIZE = 64
IMAGE_SIZE = 64
LATENT_DIM = 100
EPOCHS = 10
LR = 0.0002
BETA1 = 0.5


PROJECT_DIR = Path(__file__).resolve().parent #.parent
FAKE_DIRS = [
    PROJECT_DIR / "inpainting",
    PROJECT_DIR / "insight",
    PROJECT_DIR / "text2img",
]
OUTPUT_ROOT = PROJECT_DIR / "Image_Generation"

def get_output_dir(epochs):
    return OUTPUT_ROOT / f"epochs_{epochs}" / "dcgan_outputs"

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class FakeImageDataset(Dataset):
    def __init__(self, fake_dirs, image_size=64):
        self.paths = []

        for folder in fake_dirs:
            self.paths.extend(folder.rglob("*.jpg"))
            self.paths.extend(folder.rglob("*.jpeg"))
            self.paths.extend(folder.rglob("*.png"))

        if len(self.paths) == 0:
            raise ValueError("No fake images found.")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        return image

dataset = FakeImageDataset(FAKE_DIRS, image_size=IMAGE_SIZE)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# print(f"Number of images: {len(dataset)}")

# batch = next(iter(dataloader))
# print("Batch shape:", batch.shape)
# print("Min pixel value:", batch.min().item())
# print("Max pixel value:", batch.max().item())

# Generating the weights based on the layer needed
def weights_init(module):
    classname = module.__class__.__name__

    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3, features_g=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
    
class Discriminator(nn.Module):
    def __init__(self, channels=3, features_d=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


generator = Generator(latent_dim=LATENT_DIM).to(device)
generator.apply(weights_init)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

# because discriminator outputs probabilities
criterion = nn.BCELoss()

optimizerD = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))

def save_generated_images(images, epoch, output_dir, n=8):
    output_dir.mkdir(parents=True, exist_ok=True)

    images = images[:n].detach().cpu()
    images = (images + 1) / 2

    grid = make_grid(images, nrow=n)
    save_image(grid, output_dir / f"epoch_{epoch:03d}.png")

# save_generated_images(generated_images, epoch=0, output_dir=OUTPUT_DIR)

fixed_noise = torch.randn(8, LATENT_DIM, 1, 1, device=device)


def train_one_epoch():
    generator.train()
    discriminator.train()

    total_loss_D = 0.0
    total_loss_G = 0.0

    for dataset_images in dataloader:
        dataset_images = dataset_images.to(device)
        current_batch_size = dataset_images.size(0)

        real_labels = torch.ones(current_batch_size, device=device)
        fake_labels = torch.zeros(current_batch_size, device=device)

        noise = torch.randn(current_batch_size, LATENT_DIM, 1, 1, device=device) # we generate a new batch of noise for each batch of real images
        generated_images = generator(noise)

        optimizerD.zero_grad()

        # Evaluation on real images
        real_output = discriminator(dataset_images).view(-1)  #view: [batch, 1, 1, 1] -> [batch]
        loss_real = criterion(real_output, real_labels)

        # Evaluation on generated images
        fake_output = discriminator(generated_images.detach()).view(-1) #detach: generator's weights won't be updated during backward pass
        loss_fake = criterion(fake_output, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # Training generator
        optimizerG.zero_grad()

        output_for_generator = discriminator(generated_images).view(-1)
        loss_G = criterion(output_for_generator, real_labels) # we want the label 1 - our images should be classified as real by the discriminator

        loss_G.backward()
        optimizerG.step()

        total_loss_D += loss_D.item()
        total_loss_G += loss_G.item()

    avg_loss_D = total_loss_D / len(dataloader)
    avg_loss_G = total_loss_G / len(dataloader)

    print(f"Epoch finished | loss_D: {avg_loss_D:.4f} | loss_G: {avg_loss_G:.4f}")

    with torch.no_grad():
        preview_images = generator(fixed_noise) # i check the trained generator on fixed noise - this si previewed

    return avg_loss_D, avg_loss_G, preview_images


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCGAN generator/discriminator")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    return parser.parse_args()


def main(epochs):
    output_dir = get_output_dir(epochs)
    for epoch in range(1, epochs + 1):
        loss_D, loss_G, preview_images = train_one_epoch()

        print(f"Epoch [{epoch}/{epochs}] | loss_D: {loss_D:.4f} | loss_G: {loss_G:.4f}")

        save_generated_images(preview_images, epoch, output_dir, n=8)


if __name__ == "__main__":
    args = parse_args()
    main(args.epochs)

