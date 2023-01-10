import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()

        # ---------------Variables----------------------------
        # Data Input Config
        self.batch_size = config.batch_size  # default 128
        self.input_dim = config.input_dim  # default 3

        # Network Config
        self.input_embed = config.input_embed  # default 128
        self.num_neurons = config.hidden_dim  # default 128
        self.initializer = torch.nn.init.xavier_normal_

        # Network Modules from nn
        self.embedded_Conv1D = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.input_embed,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        nn.init.xavier_normal_(self.embedded_Conv1D.weight)
        self.normalize = nn.BatchNorm1d(self.input_embed)

        self.lstm = nn.LSTM(
            input_size=self.input_embed,
            hidden_size=self.num_neurons,
            num_layers=1,
            batch_first=False,
            bidirectional=True,
        )

    def forward(self, x):

        r"""
        x has dimention  [batch,features,len_steps] ( convulution 1D has dimention [batch,channels,len_data])
        this arrenge is changed for access to LSTM [steps_len,batch, hidden_size] with method permute.
        """
        embedded_input = self.embedded_Conv1D(x)
        embedded_input = self.normalize(embedded_input)
        embedded_input = embedded_input.permute(2, 0, 1)
        encoder_output, encoder_states = self.lstm(embedded_input)

        r"""
        the exit output is something has the for out_hiddenn and states

        out_hidden has the shape [steps,batch,features]
        hiddem estate are two values (h_state and c_state) and has the shape [num_layers*num directions, batch, hidden_size]
        """

        return encoder_output, encoder_states