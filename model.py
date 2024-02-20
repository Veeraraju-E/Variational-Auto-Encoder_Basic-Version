import torch
from torch import nn
# nn.Module documentation - https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module
# Idea: Input Img -> Encoder -> Latent Space -> Extract Mean, std -> Reparametrization trick -> Decoder -> Output
# model.py should only contain the architecture, no real logic, no loss function/ training etc


class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=200, reparam_dim=20):
        super().__init__()

        self.imgToLatent = nn.Linear(in_dim, latent_dim)
        self.latentToMu = nn.Linear(latent_dim, reparam_dim)
        self.latentToSigma = nn.Linear(latent_dim, reparam_dim)

        self.reparamToOuterLatent = nn.Linear(reparam_dim, latent_dim)
        self.outerLatentToImg = nn.Linear(latent_dim, in_dim)

        self.relu = nn.ReLU()
    # we need encoder, decoder, and forward pass methods while taking care of the dimensions

    def encoder(self, x):
        # encoder
        # it consists of input to latent space layer
        # we also need two layers, that represent the latent space to reparameterized space;
        # we would push these 2 layers, towards Gaussian distribution (this would be the KL divergence part of Loss fn)
        latentLayer = self.imgToLatent(x)
        # print(type(latentLayer))
        latentLayer = self.relu(latentLayer) # imp, not nn.ReLU(latentLayer)
        mu, sigma = self.latentToMu(latentLayer), self.latentToSigma(latentLayer)
        return mu, sigma

    def decoder(self, reparamLayer):
        # decoder
        # we have to basically reverse the layers' order as compared to the encoder, hence
        # we need reparameterized layer to the latent space layer (not the original, but just in terms of the dimensions)
        # and latent space to output layer (with same dimensions as the input)

        # reparamLayer would have both mu and sigma
        outerLatentLayer = self.reparamToOuterLatent(reparamLayer)
        outputLayer = self.outerLatentToImg(outerLatentLayer)
        return torch.sigmoid(outputLayer)

    def forward(self, x):
        # forward
        # start with obtaining the parts of the reparameterized layer from the encoder
        # then join the two parts, to form the final reparameterized layer
        # pass this final reparameterized layer into the decoder, to get the reconstructed output layer

        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(sigma)  # introduction of the Gaussian Noise
        reparamLayer = mu + epsilon*sigma  # final reparametrized layer
        output = self.decoder(reparamLayer)  # final reconstructed layer
        return output, mu, sigma  # mu and sigma would be required for loss calculation


# if __name__ == '__main__':
#     x = torch.randn(4, 784)
#     print(x)
#     vae = VariationalAutoEncoder(in_dim=784)
#     x_reconstructed, mu, sigma = vae.forward(x)
#     print(x_reconstructed)
