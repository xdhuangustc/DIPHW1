import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a pytorch tensor to a numpy array suitable for opencv.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to Numpy array
    image = tensor.cpu().detach().numpy()

    image = np.transpose(image, (1, 2, 0))

    image = (image + 1) / 2
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Curent epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatentate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True) 
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Use Tanh activation function to ensure output values are between -1 and 1
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # Decoder forward pass
        x5 = self.deconv1(x4)
        x6 = self.deconv2(x5 + x3)
        x7 = self.deconv3(x6 + x2)
        x8 = self.deconv4(x7 + x1)

        output = x8
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=3),
        )
    
    def forward(self, anno, img):
        x = torch.cat([anno, img], dim = 1)
        x = self.conv1(x)
        x = torch.sigmoid(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
discriminator = Discriminator().to(device)
generator = Generator().to(device)

optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))


train_dataset = FacadesDataset(list_file='train_list.txt')
val_dataset = FacadesDataset(list_file='val_list.txt')

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=8, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8)

loss_f1 = nn.BCELoss()
loss_f2 = nn.L1Loss()
LAMBDA = 100

num_epochs = 800
for epoch in range(num_epochs):
    loss_D = 0.0
    loss_G = 0.0
    for i, (annos, imgs) in enumerate(train_loader):
        annos = annos.to(device)
        imgs = imgs.to(device)

        optimizer_D.zero_grad()
        disc_real_output = discriminator(annos, imgs)
        d_real_loss = loss_f1(disc_real_output, torch.ones_like(disc_real_output, device=device))
        d_real_loss.backward()

        gen_output = generator(annos)
        disc_fake_output = discriminator(annos, gen_output.detach())
        d_fake_loss = loss_f1(disc_fake_output, torch.zeros_like(disc_fake_output, device=device))
        d_fake_loss.backward()

        disc_loss = d_real_loss + d_fake_loss
        optimizer_D.step()

        optimizer_G.zero_grad()
        disc_gen_output = discriminator(annos, gen_output)
        gen_crossentropy_loss = loss_f1(disc_gen_output, torch.ones_like(disc_gen_output, device=device))
        gen_l1_loss = loss_f2(gen_output, imgs)
        gen_loss = gen_crossentropy_loss + LAMBDA * gen_l1_loss
        gen_loss.backward()
        optimizer_G.step()

        loss_D += disc_loss.item()
        loss_G += gen_loss.item()

        if epoch % 5 == 0 and i == 0:
            save_images(annos, imgs, gen_output, 'train_results', epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss_D: {disc_loss.item():.4f}, Loss_G: {gen_loss.item():.4f}')
    
    avg_loss_D = loss_D / len(train_loader)
    avg_loss_G = loss_G / len(train_loader)
    print(f'Epoch: {epoch + 1}, Loss_D: {avg_loss_D:.4f}, Loss_G: {avg_loss_G:.4f}')

    if (epoch + 1) % 50 == 0:
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(generator.state_dict(), f'checkpoints/pix2pix_generator_epoch_{epoch + 1}.pth')
        torch.save(discriminator.state_dict(), f'checkpoints/pix2pix_discriminator_epoch_{epoch + 1}.pth')


