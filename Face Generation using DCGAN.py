# Wasserstein GAN with Gradient Penalty

import warnings

warnings.filterwarnings('ignore')

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import zeros, ones
from numpy.random import randn, randint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
# Define the directory of your images on Kaggle
dataset_dir = "D:\\Major Project\\Datasets\\Celebrity_Faces_Dataset"

# Get a list of all image paths in the directory
image_paths = glob.glob(os.path.join(dataset_dir, '*.jpg'))

# Considering only the first 20,000 images
image_paths = image_paths[:20000]


# Create a function to open, crop and resize images
# Create a function to open, crop and resize images
def load_and_preprocess_real_images(image_path, target_size=(64, 64)):
    # Open the image
    img = Image.open(image_path)
    # Crop 20 pixels from the top and bottom to make it square
    img = img.crop((0, 20, 178, 198))
    # Resize the image
    img = img.resize(target_size)
    # Convert to numpy array and scale to [-1, 1]
    img = np.array(img) / 127.5 - 1
    # Transpose the dimensions to [3, 64, 64] (channel first)
    img = img.transpose(2, 0, 1)
    return img


# Open, crop, and resize all images
dataset = np.array([load_and_preprocess_real_images(img_path) for img_path in image_paths])

# Convert your NumPy dataset to PyTorch tensors and move to CUDA
tensor_dataset = torch.tensor(dataset, dtype=torch.float32, device=device)

# Create a data loader using PyTorch DataLoader with num_workers set to 0
data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=128, shuffle=True, num_workers=0)

# Print dataset shape
print(dataset.shape)

# %%
# Create a subplot for the first 25 images
fig, axes = plt.subplots(6, 6, figsize=(15, 16))

for i, ax in enumerate(axes.flat):
    # Get the i-th image
    img = dataset[i].transpose(1, 2, 0)
    # Rescale the image to [0, 1] for plotting
    img_rescaled = (img + 1) / 2
    # Plot the image on the i-th subplot
    ax.imshow(img_rescaled)
    ax.axis('off')

# Add a super title
fig.suptitle('Original Dataset Preprocessed Images', fontsize=25)

plt.tight_layout()
plt.show()

# Initialize saved_images_for_epochs
saved_images_for_epochs = []


# Define your generator model using PyTorch
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Add a channel dimension to the input tensor
        x = x.view(-1, 3, 64, 64)
        return self.model(x)


# Set noise dimension for generator input
noise_dimension = 100

discriminator = Discriminator(channels=3).to(device)
generator = Generator(latent_dim=100, channels=3).to(device)


def build_gan(generator, discriminator):
    # Setting discriminator as non-trainable, so its weights won't update when training the GAN
    discriminator.trainable = False

    # Creating the GAN model
    model = nn.Sequential(
        generator,
        discriminator
    )

    # Compiling the GAN model
    optimizer = optim.Adam(model.parameters(), lr=0.0016, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    return model, optimizer, criterion


# Build GAN model
gan_model, gan_optimizer, gan_criterion = build_gan(generator, discriminator)

# Convert your NumPy dataset to PyTorch tensors and move to CUDA
dataset = torch.from_numpy(dataset).float().to(device)


# Define the data generator
def data_generator(image_paths, batch_size):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    while True:
        batch_indices = torch.randint(0, len(image_paths), (batch_size,))
        batch_paths = [image_paths[i] for i in batch_indices]

        batch_images = []
        for path in batch_paths:
            img = Image.open(path)
            img = img.crop((0, 20, 178, 198))
            img = transform(img)
            batch_images.append(img)

        real_images = torch.stack(batch_images).to(device)
        labels = torch.ones((batch_size, 1), device=device)

        # Ensure the shape is [batch_size, height, width, channels]
        real_images = real_images.permute(0, 2, 3, 1)

        yield real_images, labels


# %%
def generate_real_samples(dataset, num_samples):
    sample_indices = randint(0, dataset.shape[0], num_samples)
    X = dataset[sample_indices]
    y = ones((num_samples, 1))
    return X, y


# %%
def generate_noise_samples(num_samples, noise_dim):
    X_noise = torch.randn(num_samples, noise_dim, 1, 1, device=device)
    return torch.tensor(X_noise, dtype=torch.float32, device=device)


# %%
def generate_fake_samples(generator, noise_dim, num_samples):
    X_noise = generate_noise_samples(num_samples, noise_dim)
    X = generator(X_noise)
    y = zeros((num_samples, 1))
    return X, y


# %%
def generate_images(epoch, generator, num_samples=6, noise_dim=100):
    """
    Generate images from the generator model for a given epoch.
    """
    generator.eval()

    with torch.no_grad():
        # Generate noise samples
        X_noise = generate_noise_samples(num_samples, noise_dim)
        X_noise = torch.tensor(X_noise, dtype=torch.float32, device=device)

        # Use generator to produce images from noise
        X = generator(X_noise).detach().cpu()

    generator.train()

    # Rescale images to [0, 1] for visualization
    X = (X + 1) / 2

    return X


# %%
def display_saved_images(saved_images, display_frequency=50):
    """
    Display saved images at a specified frequency during training.
    """
    for i, img in enumerate(saved_images):
        if i % display_frequency == 0:
            plt.figure(figsize=(8, 8))
            for j in range(len(img)):
                plt.subplot(1, len(img), j + 1)
                # Transpose and rearrange dimensions for correct display
                img_display = np.transpose(img[j], (1, 2, 0))
                plt.imshow(img_display)
                plt.axis('off')
            plt.show()


# %%
def plot_generated_images(epoch, generator, num_samples=6, noise_dim=100, figsize=(15, 3)):
    """
    Plot and visualize generated images from the generator model for a given epoch.
    """

    # Generate noise samples
    X_noise = generate_noise_samples(num_samples, noise_dim)

    # Use generator to produce images from noise
    X = generator(X_noise)

    # Move the generated images tensor to CPU
    X = X.cpu().detach()

    # Rescale images to [0, 1] for visualization
    X = (X + 1) / 2

    # Apply the desired normalization
    transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    X = transform(X)

    # Plotting the images
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)

    for i in range(num_samples):
        axes[i].imshow(np.transpose(X[i], (1, 2, 0)))
        axes[i].axis('off')

    # Add a descriptive title
    fig.suptitle(f"Generated Images at Epoch {epoch + 1}", fontsize=22)
    plt.tight_layout()
    plt.show()


# %%
def train(generator, discriminator, gan_optimizer, gan_criterion, data_loader, num_epochs=100, batch_size=128,
          display_frequency=10):
    saved_images_for_epochs = []

    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            if isinstance(data, (list, tuple)):
                real_images, labels = data[0], None
                if len(data) == 2:
                    labels = data[1].to(device) if data[1] is not None else None
            else:
                real_images = data[0].to(device)
                labels = data[1].to(device)

            batch_size = real_images.size(0)

            # Move real_images to the device
            real_images = real_images.to(device)

            # Add this print statement
            print(f"Batch {i}, Real Data shape: {real_images.shape}")

            # Generate noise samples and their corresponding labels for training the generator
            gan_noise = torch.randn(batch_size, noise_dimension, 1, 1, device=device)
            gan_labels = torch.ones((batch_size, 1), device=device)

            # Train the discriminator
            discriminator.zero_grad()
            real_output = discriminator(real_images)

            fake_images = generator(gan_noise)
            fake_output = discriminator(fake_images.detach())
            fake_loss = gan_criterion(fake_output, gan_labels)

            discriminator_loss = (real_output + fake_loss) / 2
            discriminator_loss.backward()
            gan_optimizer.step()

            # Train the generator
            generator.zero_grad()
            generated_images = generator(gan_noise)
            gan_output = discriminator(generated_images)
            generator_loss = gan_criterion(gan_output, gan_labels)

            generator_loss.backward()
            gan_optimizer.step()

            if i % display_frequency == 0:
                print(f"[Epoch {epoch + 1}/{num_epochs}] [Batch {i}/{len(data_loader)}] [D loss: {discriminator_loss.item()}] [G loss: {generator_loss.item()}]")

        # Display generated images at the specified frequency
        if epoch % display_frequency == 0:
            generated_images_for_epoch = generate_images(epoch, generator)
            saved_images_for_epochs.append(generated_images_for_epoch)

            # Plot generated images to visualize the progress of the generator
            plot_generated_images(epoch, generator)

    return saved_images_for_epochs


# %%
# Convert your NumPy dataset to PyTorch tensors and move to CUDA
tensor_dataset = torch.tensor(dataset, dtype=torch.float32, device=device)

# Create a data loader using PyTorch DataLoader with num_workers set to 0
data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=128, shuffle=True, num_workers=0)

# Train the GAN model on the dataset
saved_images = train(generator, discriminator, gan_optimizer, gan_criterion, data_loader, num_epochs=250,
                     batch_size=128, display_frequency=5)

# Display all the saved images during training
display_saved_images(saved_images, display_frequency=5)


# %%
def plot_generated_images_after_training(generator, noise_dim=100, figsize=(15, 16)):
    fig, axes = plt.subplots(6, 6, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        # Generate noise samples
        X_noise = generate_noise_samples(1, noise_dim)

        # Use generator to create an image
        X = generator(X_noise)

        # Rescale images to [0, 1] for plotting
        X = (X + 1) / 2

        # Plot the image on the i-th subplot
        ax.imshow(X[0])
        ax.axis('off')

    # Add a super title
    fig.suptitle('Generated Images after Training for 250 Epochs', fontsize=25)

    plt.tight_layout()
    plt.show()


# Display generated images after training
plot_generated_images_after_training(generator)
