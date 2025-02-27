import torch
from .vanilla_vae import VanillaVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ConditionalVAE(VanillaVAE):

    def __init__(self,
                 in_channel: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 num_classes: int = 11,
                 img_size:int = 320,
                 **kwargs) -> None:
        super().__init__(in_channel=in_channel+1,latent_dim=latent_dim,hidden_dims=hidden_dims,out_channel=in_channel,**kwargs)

        self.latent_dim = latent_dim
        self.img_size = img_size

        self.embed_class = nn.Linear(num_classes, img_size)
        self.embed_data = nn.Conv1d(in_channel, in_channel, kernel_size=1)

        self.decoder_input = nn.Linear(latent_dim + num_classes, self.hidden_dims[0] * self.encoded_width)
        self.num_classes = num_classes


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = nn.functional.one_hot(kwargs['labels'],self.num_classes).float()
        embedded_class = self.embed_class(y).unsqueeze(1)

        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim = 1)

        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, y], dim = 1)
        return  [self.decode(z), input, mu, log_var]

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = nn.functional.one_hot(kwargs['labels'],num_classes=self.num_classes).float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

