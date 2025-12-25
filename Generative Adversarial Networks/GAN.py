import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# ------------------------ARCHITECTURES--------------------------

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        return self.discriminator(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.generator(x)


# -----------------------HYPER PARAMETERS----------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 64
img_dim = 28 * 28 * 1 # mnist dimensions
learning_rate = 3e-4
num_epochs = 50
batch_size = 32

discriminator = Discriminator(img_dim).to(device)
generator = Generator(z_dim, img_dim).to(device)

input_noise = torch.randn((batch_size, z_dim)).to(device)

mean = (0.5,)
std = (0.5,)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

dataset = datasets.MNIST(root='dataset/', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

criterion = nn.BCEWithLogitsLoss()

writer_fake = SummaryWriter("runs/GAN_MNIST/fake")
writer_real = SummaryWriter("runs/GAN_MNIST/real")
step = 0

# ----------------------TRAINING LOOP-------------------------

for epoch in range(num_epochs):
    for batch, (real_img, _) in enumerate(dataloader):
        real_img = real_img.view(-1, 784).to(device)
        epoch_batch_size = real_img.shape[0]
        
        noise = torch.randn(epoch_batch_size, z_dim).to(device)
        fake_img = generator(noise)
        
        # Train Discriminator
        discriminator_real = discriminator(real_img).view(-1)
        # print("discriminator_real: ", discriminator_real)
        
        discriminator_fake = discriminator(fake_img.detach()).view(-1)
        # print("discriminator_fake: ", discriminator_fake)

        # max log(D(real)) + log(1 - D(fake))
        disc_loss_real = criterion(discriminator_real, torch.ones_like(discriminator_real))
        disc_loss_fake = criterion(discriminator_fake, torch.zeros_like(discriminator_fake))
        
        discriminator_loss = (disc_loss_real + disc_loss_fake) / 2
        
        discriminator.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()
        
        # Train Generator
        # min log(1 - D(G(z))) -> max log(D(G(z)))
        
        discriminator_output = discriminator(fake_img).view(-1)
        generator_loss = criterion(discriminator_output, torch.ones_like(discriminator_output))
        
        generator.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()
        
        # Tensorboard config
        if batch == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] | Loss D: {discriminator_loss:.4f}, Loss G: {generator_loss:.4f}"
            )
            
            with torch.no_grad():
                fake = generator(input_noise).reshape(-1, 1, 28, 28)
                data = real_img.reshape(-1, 1, 28, 28)

                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                
                writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("MNIST Real Images", img_grid_real, global_step=step)
                
                step += 1
                
input("Training finished. Press Enter to exit...")
                
writer_fake.close()                
writer_real.close()                