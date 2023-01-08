import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import logging


class Critic(nn.Module):
    def __init__(self, config, device):
        super(Critic, self).__init__()
        self.config = config
        self.device = device
        self.mode = config.mode

        # embbing network
        self.num_neurons = config.hidden_dim
        self.embedding = Critic_emb(self.config)

        # Attention mecanism with glimpse
        # Cada entrada es multiplidcada por una entrada
        self.attention_Wref = nn.Conv1d(
            in_channels=self.num_neurons * 2,
            out_channels=self.num_neurons * 2,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.attention_Wq = nn.Conv1d(
            in_channels=self.num_neurons * 2,
            out_channels=self.num_neurons * 2,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.attention_v = nn.Conv1d(
            in_channels=self.num_neurons * 2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.Linear_1 = nn.Linear(
            in_features=self.num_neurons * 2,
            out_features=self.num_neurons * 2,
            bias=True,
        )
        self.Linear_2 = nn.Linear(
            in_features=self.num_neurons * 2, out_features=1, bias=True
        )

        # first element linear layer
        self.linear_first_entry = nn.Linear(
            in_features=2, out_features=self.num_neurons * 2, bias=True
        )

        """
        probablemente así con una convolución logre el mismo resultado

        self.attention_Wq = torch.empty(size= (self.num_neurons,self.neurons), dtype = torch.float32)
        nn.init.xavier_normal_(self.attention_Wq)
        self.attention_Wq = nn.Paramer(self.attention_Wq)
        """

    def forward(self, x, kwargs):

        encoder_output, encoder_states = self.embedding(x)

        batch_size = encoder_output.shape[1]

        c_state = (
            encoder_states[1]
            .permute(1, 2, 0)
            .reshape(batch_size, self.num_neurons * 2, 1)
        )  # (1,batch, hidden_dim) to (batch, hidden_dim, 1)

        first_entry = self.first_entry(batch_size, kwargs)
        encoder_output = torch.cat((first_entry, encoder_output), dim=0)

        # (steps, batch, hidden_dim) to (batch, hidden_dim, steps)
        arrange_input = encoder_output.permute(1, 2, 0)
        Wref_dot = self.attention_Wref(arrange_input)  # (batch, channels, len)
        Wq_dot = self.attention_Wq(c_state)  # (batch, hidden_dim, 1)
        scores = self.attention_v(torch.tanh(Wref_dot + Wq_dot))
        ponderate = F.softmax(scores, dim=2)  # (batch, 1 ,7)
        # glimpse
        glimse = arrange_input * ponderate  # (batch, hidden_dim, steps)

        glimse = torch.sum(glimse, dim=2)  # (batch, hidden_dim)
        # print(glimse.shape)

        linear = self.Linear_1(glimse)
        linear = self.Linear_2(linear)

        return linear

    def first_entry(self=None, batchSize=None, kwargs=None):

        # Fist entry with  static values
        first_element = (
            torch.tensor([1.1, 1.4], dtype=torch.float32)
            .repeat(batchSize, 1)
            .to(self.device)
        )

        if self.mode == "n" or self.mode == "k":
            if self.mode == "n":
                restricted_data = kwargs["restricted_n"]
            elif self.mode == "k":
                restricted_data = kwargs["restricted_k"]

            first_element[:, 1] = 1 / restricted_data

        first_element = self.linear_first_entry(first_element).unsqueeze(dim=0)
        first_element = torch.tanh(first_element)

        return first_element


class Critic_emb(nn.Module):
    def __init__(self, config):
        super(Critic_emb, self).__init__()

        self.config = config
        self.input_embed = config.input_embed
        self.input_dim = config.input_dim
        self.num_neurons = config.hidden_dim

        self.embed_1D = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.input_embed,
            kernel_size=1,
            stride=1,
        )
        self.normalize = nn.BatchNorm1d(self.input_embed)

        self.lstm = nn.LSTM(
            input_size=self.input_embed,
            hidden_size=self.num_neurons,
            num_layers=1,
            batch_first=False,
            bidirectional=True,
        )

    def forward(self, x):
        embedded_input = self.embed_1D(x)
        embedded_input = self.normalize(embedded_input)
        embedded_input = embedded_input.permute(2, 0, 1)
        encoder_output, encoder_states = self.lstm(embedded_input)

        return encoder_output, encoder_states
