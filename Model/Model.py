import torch.nn as nn
from Model.MaskNets import MultiNets, Generator
from Model.MLP import Classifier

# Model Training with Collaborative Dual-Task Learning
class CDTL(nn.Module):
    def __init__(self, model_config):
        super(CDTL, self).__init__()
        self.data_dim = model_config['data_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.z_dim = model_config['z_dim']
        self.mask_num = model_config['mask_num']
        self.en_nlayers = model_config['en_nlayers']
        self.de_nlayers = model_config['de_nlayers']
        self.mask_model = Generator(MultiNets(), model_config)
        self.ssl_model = Classifier(self.z_dim, self.hidden_dim, self.z_dim)

        encoder = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers-1):
            encoder.append(nn.Linear(encoder_dim,self.hidden_dim,bias=False))
            encoder.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim

        encoder.append(nn.Linear(encoder_dim,self.z_dim,bias=False))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        decoder_dim = self.z_dim
        for _ in range(self.de_nlayers-1):
            decoder.append(nn.Linear(decoder_dim,self.hidden_dim,bias=False))
            decoder.append(nn.LeakyReLU(0.2, inplace=True))
            decoder_dim = self.hidden_dim

        decoder.append(nn.Linear(decoder_dim,self.data_dim,bias=False))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x_input):
        x_mask, masks = self.mask_model(x_input)
        x_multi = x_input.unsqueeze(1).repeat(1, self.mask_num, 1)
        B, T, D = x_mask.shape
        x_mask = x_mask.reshape(B*T, D)
        x_multi = x_multi.reshape(B * T, D)
        z_x = self.encoder(x_multi)
        z = self.encoder(x_mask)

        z_x_pred = self.ssl_model(z_x)
        x_pred = self.decoder(z)

        z = z.reshape(x_input.shape[0], self.mask_num, z.shape[-1])
        x_pred = x_pred.reshape(x_input.shape[0], self.mask_num, x_input.shape[-1])
        z_x = z_x.reshape(x_input.shape[0], self.mask_num, z_x.shape[-1])
        z_x_pred = z_x_pred.reshape(x_input.shape[0], self.mask_num, z_x_pred.shape[-1])

        return x_pred, z, masks, z_x, z_x_pred