import torch
from torch import nn
import itertools
import logging
from scipy.special import factorial

# ---------- local modules ------------
from pointer import Pointer
from actor.embedding import Embedding

# this function was disable by ask of the dr
# from batchGenerator import optimisticPessimistic 

# import matplotlib.pyplot as plt


class Actor(nn.Module):
    def __init__(self, config, device):
        super(Actor, self).__init__()

        # ---------------Variables----------------------------

        self.config = config
        self.device = config.device
        self.log = logging.getLogger(__name__)

        # Data Input Config
        self.batch_size = config.batch_size  # default 128
        self.max_size = config.max_length  # default 7
        self.input_dim = config.input_dim  # default 3

        # Network Config
        self.input_embed = config.input_embed  # default 128
        self.num_neurons = config.hidden_dim  # default 128
        self.initializer = torch.nn.init.xavier_normal_
        self.extraElements = config.extraElements

        # Reward config
        self.beta = config.beta

        # -----------   Networks  -----------------------------------
        self.embedding_layer = Embedding(self.config)
        self.decoder_pointer = Pointer(self.config, device)

        # ------=----- Dummies objects for selections---------
        self.Linear_dummies = nn.Linear(
            in_features=4, out_features=self.num_neurons * 2, bias=True
        )

    def forward(self=None, x=None, kwargs=None):
        # x(batch), dims = [batch_len, dimention, clouds_qnt]
        self.batch_qnt = x.shape[0]
        self.parameters = x.shape[1]
        self.log.info(
            "Dimenciíon del batch es al cantidad de iteracines que tendra que hacer: %s",
            str(x.shape[2]),
        )

        # Encoder_part
        encoder_output, encoder_states = self.embedding_layer(x)

        if self.extraElements == "afterDecoder":
            # Stop_condition and k selection (concatenation)... 2 dummies
            if self.config.mode == "k":
                raise NotImplementedError("not implemented for mode = k")
            dummies = self.extraPositions()
            encoder_output = torch.cat((encoder_output, dummies), dim=0)

        # Decoder part
        selections, log_probs = self.decoder_pointer(
            encoder_output, encoder_states, kwargs
        )

        return selections, log_probs

    def extraPositions(self):
        dummy_k = torch.ones(
            size=(self.batch_qnt, 4),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        dummy_end_k = (
            torch.ones(
                size=(self.batch_qnt, 4),
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            * 0.5
        )

        dummy_k = self.Linear_dummies(dummy_k)
        dummy_k = torch.tanh(dummy_k)
        dummy_k = dummy_k.unsqueeze(dim=0)

        dummy_end_k = self.Linear_dummies(dummy_end_k)
        dummy_end_k = torch.tanh(dummy_end_k)
        dummy_end_k = dummy_end_k.unsqueeze(dim=0)

        dummies = torch.cat((dummy_k, dummy_end_k), dim=0)

        return dummies