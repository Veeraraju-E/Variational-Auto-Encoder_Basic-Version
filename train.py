import torch
from torch import nn, optim
import torchvision.datasets as datasets  # for the MNIST dataset
from torchvision import transforms  # for the augmentations
from torchvision.utils import save_image
from tqdm import tqdm   # for that progress bar
from torch.utils.data import DataLoader  # dataset management (batch sizes blah blah)
from model import VariationalAutoEncoder

# Initial Constants
DEVICE = torch.device('cpu')
EPOCHS = 10
LR = 1e-2  # 3e-4 is the Karpathy Constant
BATCH_SIZE = 32
IN_DIM = 28*28
LATENT_DIM = 200
REPARAM_DIM = 20

# Dataset Loading
dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# the model
model = VariationalAutoEncoder(in_dim=IN_DIM, latent_dim=LATENT_DIM, reparam_dim=REPARAM_DIM).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss(reduction='sum')


def train(loader=train_dataloader, model=model):
    for epoch in range(EPOCHS):
        loop = tqdm(enumerate(loader))

        for batch_idx, (x, _) in loop:  # we are not interested in the labels
            x = x.to(device=DEVICE)
            x = x.reshape(x.shape[0], -1)  # convert [32, 1, 28, 28] to [32, 784]

            # forward
            x_reconstructed, mu, sigma = model.forward(x)
            reconstruction_loss = loss_fn(x_reconstructed, x)
            # kl_divergence = torch.sum(1 + log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) -> maximize this or minimize the -
            kl_divergence = -0.5*torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))  # push towards Normal distribution
            loss = reconstruction_loss + kl_divergence  # keep the individual weights as 1 itself

            # backward prop
            optimizer.zero_grad()
            loss.backward()

            # Adam
            optimizer.step()

            loop.set_postfix(loss=loss.item())


def generate(digit, num_samples):
    # it takes in digit -> the digit to be generated, and also the number of such samples

    # let's first get one image corresponding to each digit, and use this as the base, bu putting it into a list
    idx = 0
    images_base = []
    for (x, y) in dataset:
        if y == idx:
            x = x.reshape(1, 784)
            images_base.append(x)
            idx += 1
        if idx == 10:
            break

    # now we need to create the reparameterized layer for this digit, for this we need the mu, sigma
    mu, sigma = None, None
    with torch.no_grad():
        mu, sigma = model.encoder(images_base[digit])

    print(mu.shape, sigma.shape)

    for sample in range(num_samples):
        epsilon = torch.randn_like(sigma)
        reparamLayer = mu + epsilon*sigma
        output = model.decoder(reparamLayer)
        output = output.reshape(-1, 1, 28, 28)  # expand it back up
        save_image(output, f'generated_{digit}_ex{sample}.png')


if __name__ == '__main__':
    train()
    for idx in range(10):
        generate(idx, num_samples=4)
