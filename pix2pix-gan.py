import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from glob import glob
from keras.src.utils import img_to_array, load_img
from keras_core.src.ops.image import resize
import numpy as np
import matplotlib.pyplot as plt
import time


# Data Loading Functions
def load_image(path, SIZE):
    img = torch.from_numpy(
        np.transpose(resize(img_to_array(load_img(path)) / 255., (SIZE, SIZE)).numpy(), (2, 0, 1))
    ).float()
    return img


def load_images(paths, SIZE=256):
    images = []
    for path in tqdm(paths, desc="Loading"):
        img = load_image(path, SIZE)
        images.append(img)

    print("Number of loaded images:", len(images))

    if len(images) > 0:
        return torch.stack(images)
    else:
        raise ValueError("No images loaded!")


# Image Paths
image_paths = sorted(glob("D:\\Major Project\\Datasets\\celeb\\img_align_celeba\\celeb dataset" + "/*.jpg"))
print(f"Total Number of Images: {len(image_paths)}")

# Limiting the number of images for faster training
images = load_images(image_paths[:10000])


# Define the Generator and Discriminator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Instantiate models
generator = Generator()
discriminator = Discriminator()

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Move data to GPU
images = images.to(device)

# Define your loss functions and optimizers
criterion = nn.BCELoss()

optimizer_gen = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_dis = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Training loop
num_epochs = 250
save_interval = 10  # Save generated image every 10 epochs
total_batches = len(images) // 32

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    start_time = time.time()

    # Create tqdm progress bar for batches
    with tqdm(total=total_batches, unit="batch") as pbar:
        for batch_idx, X_b in enumerate(DataLoader(images, batch_size=32, shuffle=True)):
            real_labels = torch.ones(X_b.size(0), 1, 32, 32, device=device)
            fake_labels = torch.zeros(X_b.size(0), 1, 32, 32, device=device)

            X_b = X_b.to(device)

            noise = torch.randn_like(X_b).to(device)
            gen_out = generator(noise)

            # Train Discriminator
            discriminator.train()
            optimizer_dis.zero_grad()
            dis_real = discriminator(X_b)
            dis_fake = discriminator(gen_out.detach())
            loss_dis_real = criterion(dis_real, real_labels)
            loss_dis_fake = criterion(dis_fake, fake_labels)
            loss_dis = (loss_dis_real + loss_dis_fake) / 2.0
            loss_dis.backward()
            optimizer_dis.step()

            # Train Generator
            generator.train()
            optimizer_gen.zero_grad()
            dis_fake = discriminator(gen_out)
            loss_gen = criterion(dis_fake, real_labels)
            loss_gen.backward()
            optimizer_gen.step()

            # Update progress bar
            pbar.set_postfix({"Discriminator Loss": loss_dis.item(), "Generator Loss": loss_gen.item()})
            pbar.update(1)

    # Calculate epoch duration
    epoch_duration = time.time() - start_time
    remaining_time = epoch_duration * (num_epochs - epoch - 1)
    print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds. Remaining time: {remaining_time:.2f} seconds")

    # Save generated image every 10 epochs
    if (epoch + 1) % save_interval == 0:
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(1, 3, 256, 256).to(device)
            gen_face = generator(noise).cpu().numpy()[0]

        plt.imshow(np.transpose(gen_face, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(f"generated_image_epoch_{epoch + 1}.png")
        plt.close()

# Save the trained models after completing the training
torch.save(generator.state_dict(), "Pix2Pix_Generator.pth")
torch.save(discriminator.state_dict(), "Pix2Pix_Discriminator.pth")

# Generate images using the trained generator
generator.eval()
with torch.no_grad():
    noise = torch.randn(1, 3, 256, 256).to(device)
    gen_face = generator(noise).cpu().numpy()[0]

plt.imshow(np.transpose(gen_face, (1, 2, 0)))
plt.axis('off')
plt.show()