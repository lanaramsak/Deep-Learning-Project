# Upgraded DCGAN script
# Changes from the original:
# - Discriminator uses spectral normalization for more stable GAN training
# - Discriminator and generator both use larger feature maps (128 base channels)
# - Output directory is still organized by epoch count
# - Maintains the same training loop style as the original script
# - Uses hinge loss and instance noise for a stronger adversarial objective

# More changes
# - features_d 128 -> 64 to reduce model capacity and encourage more stable training

import random
from pathlib import Path
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import argparse

SEED = 42
BATCH_SIZE = 64
IMAGE_SIZE = 64
LATENT_DIM = 100
EPOCHS = 10
LR = 0.0002
BETA1 = 0.5
INSTANCE_NOISE_STD = 0.1

PROJECT_DIR = Path(__file__).resolve().parent
FAKE_DIRS = [
    PROJECT_DIR / "inpainting",
    PROJECT_DIR / "insight",
    PROJECT_DIR / "text2img",
]
OUTPUT_ROOT = PROJECT_DIR / "Image_Generation"


def get_run_dir(epochs, run_name=""):
    base_name = f"upgrade_epochs_{epochs}"
    if run_name:
        base_name = f"{base_name}_{run_name}"
    return OUTPUT_ROOT / base_name


def get_output_dir(epochs, run_name=""):
    return get_run_dir(epochs, run_name) / "dcgan_outputs"


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
            transforms.Resize(image_size + 8),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        return image


def weights_init(module):
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3, features_g=128):
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
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, channels=3, features_d=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, features_d, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False)),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


generator = Generator(latent_dim=LATENT_DIM).to(device)
generator.apply(weights_init)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

optimizerD = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(BETA1, 0.999))
optimizerG = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(BETA1, 0.999))


def save_generated_images(images, epoch, output_dir, n=8):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = images[:n].detach().cpu()
    images = (images + 1) / 2
    grid = make_grid(images, nrow=n)
    save_image(grid, output_dir / f"epoch_{epoch:03d}.png")


def append_metrics(run_dir, epoch, loss_d, loss_g, instance_noise_std):
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "training_metrics.csv"
    file_exists = metrics_path.exists()
    with metrics_path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["epoch", "loss_D", "loss_G", "instance_noise_std"])
        writer.writerow([epoch, f"{loss_d:.6f}", f"{loss_g:.6f}", f"{instance_noise_std:.6f}"])


def current_instance_noise_std(epoch, total_epochs, base_std):
    if total_epochs <= 1:
        return 0.0
    progress = (epoch - 1) / (total_epochs - 1)
    return max(0.0, base_std * (1.0 - progress))


def add_instance_noise(images, std):
    if std <= 0.0:
        return images
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, -1.0, 1.0)


fixed_noise = torch.randn(8, LATENT_DIM, 1, 1, device=device)


def train_one_epoch(epoch, total_epochs, base_instance_noise_std):
    generator.train()
    discriminator.train()

    total_loss_D = 0.0
    total_loss_G = 0.0
    instance_noise_std = current_instance_noise_std(epoch, total_epochs, base_instance_noise_std)

    for dataset_images in dataloader:
        dataset_images = dataset_images.to(device)
        current_batch_size = dataset_images.size(0)

        noise = torch.randn(current_batch_size, LATENT_DIM, 1, 1, device=device)
        generated_images = generator(noise)
        real_images_noisy = add_instance_noise(dataset_images, instance_noise_std)
        fake_images_noisy = add_instance_noise(generated_images.detach(), instance_noise_std)

        optimizerD.zero_grad()
        real_output = discriminator(real_images_noisy).view(-1)
        loss_real = F.relu(1.0 - real_output).mean()

        fake_output = discriminator(fake_images_noisy).view(-1)
        loss_fake = F.relu(1.0 + fake_output).mean()

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        optimizerG.zero_grad()
        generated_images_noisy = add_instance_noise(generated_images, instance_noise_std)
        output_for_generator = discriminator(generated_images_noisy).view(-1)
        loss_G = -output_for_generator.mean()

        loss_G.backward()
        optimizerG.step()

        total_loss_D += loss_D.item()
        total_loss_G += loss_G.item()

    avg_loss_D = total_loss_D / len(dataloader)
    avg_loss_G = total_loss_G / len(dataloader)

    print(f"Epoch finished | loss_D: {avg_loss_D:.4f} | loss_G: {avg_loss_G:.4f}")
    with torch.no_grad():
        preview_images = generator(fixed_noise)
    return avg_loss_D, avg_loss_G, preview_images, instance_noise_std


def parse_args():
    parser = argparse.ArgumentParser(description="Train upgraded DCGAN generator/discriminator")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional suffix for the output directory, e.g. bn_restore",
    )
    parser.add_argument(
        "--instance-noise-std",
        type=float,
        default=INSTANCE_NOISE_STD,
        help=f"Initial std for instance noise, linearly decayed to 0 by the last epoch (default: {INSTANCE_NOISE_STD})",
    )
    return parser.parse_args()


def main(epochs, run_name, base_instance_noise_std):
    run_dir = get_run_dir(epochs, run_name)
    output_dir = get_output_dir(epochs, run_name)
    print(f"Running on device: {device}")
    print(f"Dataset size: {len(dataset)} images")
    print(f"Saving outputs to: {run_dir}")
    for epoch in range(1, epochs + 1):
        loss_D, loss_G, preview_images, instance_noise_std = train_one_epoch(
            epoch, epochs, base_instance_noise_std
        )
        print(
            f"Epoch [{epoch}/{epochs}] | loss_D: {loss_D:.4f} | "
            f"loss_G: {loss_G:.4f} | instance_noise_std: {instance_noise_std:.4f}"
        )
        save_generated_images(preview_images, epoch, output_dir, n=8)
        append_metrics(run_dir, epoch, loss_D, loss_G, instance_noise_std)


if __name__ == "__main__":
    args = parse_args()
    dataset = FakeImageDataset(FAKE_DIRS, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    main(args.epochs, args.run_name, args.instance_noise_std)
