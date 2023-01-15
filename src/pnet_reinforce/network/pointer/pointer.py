import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import logging

from network.pointer.mask import pointer_mask, mask_n, mask_k
from generator.data_interface.data import DataRepresentation


class Pointer(nn.Module):
    def __init__(self, config, device):
        super(Pointer, self).__init__()
        self.device = device
        self.config = config
        # masks
        self.pointer_mask = pointer_mask
        self.mask_k = mask_k
        self.mask_n = mask_n
        # Mask modes according with model (mono objetive or multi objetive)
        self.mode = config.mode
        self.mode_mask = {"k_n": self.pointer_mask, "n": self.mask_n, "k": self.mask_k}
        self.log = logging.getLogger(__name__)
        # ---------------Variables----------------------------

        # Data Input Config
        self.batch_size = config.batch_size  # default 128
        self.max_size = config.max_length  # default 7

        # Network Config
        self.hidden_dim = config.hidden_dim  # default 128 (num_neurons)
        self.initializer = torch.nn.init.xavier_normal_

        self.inference_mode = not config.train_mode
        self.C = config.C
        self.temperature = config.temperature

        # Vector para las confuguraciones
        self.linear_first_entry = torch.nn.Linear(
            in_features=2, out_features=self.hidden_dim * 2, bias=True
        )

        # Decode layer
        self.lstm_decoder = nn.LSTM(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=False,
            bidirectional=True,
        )

        # Attention  mechanism Weights for glimpse part (def pointer mechanism)
        self.Wref_g = nn.Parameter(
            torch.empty(
                size=(self.hidden_dim * 2, self.hidden_dim * 2), dtype=torch.float32
            )
        )
        nn.init.xavier_normal_(self.Wref_g)

        self.Wq_g = nn.Parameter(
            torch.empty(
                size=(self.hidden_dim * 2, self.hidden_dim * 2), dtype=torch.float32
            )
        )
        nn.init.xavier_normal_(self.Wq_g)

        self.vector_vg = nn.Parameter(
            torch.rand(size=(self.hidden_dim * 2,), dtype=torch.float32)
        )

        # Attention mechanism Weights  for Pointer Mechanism

        self.Wref = nn.Parameter(
            torch.empty(
                size=(self.hidden_dim * 2, self.hidden_dim * 2), dtype=torch.float32
            )
        )
        nn.init.xavier_normal_(self.Wref)

        self.Wq = nn.Parameter(
            torch.empty(
                size=(self.hidden_dim * 2, self.hidden_dim * 2), dtype=torch.float32
            )
        )
        nn.init.xavier_normal_(self.Wq)

        self.vector_v = nn.Parameter(
            torch.rand(size=(self.hidden_dim * 2,), dtype=torch.float32)
        )

    def forward(self, encoder_output, encoder_states, data_object: DataRepresentation):

        r"""
        Encoder_output: [steps, batch, hidden_dim*directions]
        Encoder_states:
            h_state:[num_layers * num_directions, batch, hidden_size]
            s_tate: [num_layers * num_directions, batch, hidden_size]

        This two values are in array with first dimention equal 2

        """
        # This will we the ref matrix in the pointer mechanism
        self.encoder_output = encoder_output
        self.selections = []  # Selection are stored here

        # Here will be saved the logprobabilities of every selection
        self.log_probs = []
        # Dimentions of input tensor
        # stops_qnt is the quantity of elements to decode with the pointer
        self.steps_qnt = encoder_output.shape[0]
        self.batch_size = encoder_output.shape[1]

        # Decoder context (it's internal state of coder)
        # h_state:[num_layers * num_directions, batch, hidden_size]
        dec_s = encoder_states

        # First input in the decoder phase
        dec_ipt = self.first_entry(self.batch_size, data_object).detach()

        # Masking tensor to avoid select again elements previously
        # selected
        self.mask = torch.zeros(
            size=(self.batch_size, self.steps_qnt), device=self.device
        )
        if self.config.variable_length:
            for i, len_element in enumerate(data_object.elements_length):
                if self.config.mode == "k":
                    # elements extra(padding) are set to 1 for be non selectionables
                    self.mask[i, len_element:-1] = 1

                else:
                    self.mask[i, len_element:-2] = 1

        if self.mode == "k_n" or self.mode == "n":
            # In the coding elements, there are 2 extra elements and 'k' is subject to at least 2
            totalIter = (self.steps_qnt - 2) * 2 - 2
        elif self.mode == "k":
            # Here don't worry to select k elements only for original elements in the batch,
            # only exclude the stop element and that all
            totalIter = self.steps_qnt - 1
        else:
            raise ValueError("invalid value in config.mode")

        # Decodification cicle
        for step in range(totalIter):
            dec_ipt, dec_s = self.__decode_loop(dec_ipt, dec_s, step, data_object=data_object)

        # Secuences ordered in files [batch,secuence]
        self.selections = torch.stack(self.selections, dim=1)  # [batch,steps_secuence]

        # secuence de log sum over the steps
        self.log_probs = torch.stack(self.log_probs)  # [steps,batch]
        self.log_probs = torch.sum(self.log_probs, dim=0)  # [batch] steps dim is reduce
        self.log.info("Selections: \n%s", str(self.selections))
        self.log.info("Log probs: \n%s", str(self.log_probs))

        return self.selections, self.log_probs  # lo demás luego lo verifico

    def __decode_loop(self, decoder_loop_input, decoder_loop_state, step, data_object: DataRepresentation):

        # Run the cell on a combination of the previous input and state
        output, state = self.lstm_decoder(decoder_loop_input, decoder_loop_state)

        # Pointer mechanism to genrate the probs of every selection
        vector_pointer = self.__pointer_mechanism(
            self.encoder_output, output, step, data_object=data_object
        )  # sel_n in forward

        # Multinomial distribution
        #   Categocal distribution helps to explore different solutions over time
        #   arg.max es more aggressive without exploration, this could lead in poor solutions
        distribution_toNext_selection = Categorical(vector_pointer)

        if self.config.selection == "categorical":

            next_selection = distribution_toNext_selection.sample()
            next_selection = next_selection.type(torch.int64)

        elif self.config.selection == "argmax":
            next_selection = torch.argmax(vector_pointer, dim=1)

        else:
            raise ValueError("config.selection value no valid")

        self.selections.append(next_selection)

        # log_pron for backprob defined for REINFORCE

        self.log_probs.append(distribution_toNext_selection.log_prob(next_selection))

        # update current selection and mask for funtion attention
        self.current_selection = F.one_hot(next_selection, self.steps_qnt).to(
            self.device
        )
        self.mask = (
            self.mask.to(self.device) + self.current_selection
        )  # la k y la stop tendran muchas sumas

        tensor_len_batch = torch.tensor(range(self.batch_size), dtype=torch.int64)

        new_decoder_input = self.encoder_output[
            next_selection, tensor_len_batch
        ]  # revisar las variables
        new_decoder_input = torch.unsqueeze(new_decoder_input, dim=0)

        return new_decoder_input, state

    def __pointer_mechanism(self, ref_g, q_g, step, data_object: DataRepresentation):
        r"""
        Wref_g,W_q \in R^d*d  and u_vector = ref \in R^d

        ref_dot:
        At the begining de ref_q must have [steps, batch, hidden_dim] later
        is permuted to size [batch,hidden_dim,steps] then multiplied by Wref_g
        finally has a shape [batch,hidden_dim, steps]

        q_dot:
        This q_g must to has a dimention [batch, hidden_dim] later is permuted
        to [hidden_dim, batch], then mutiplied by W_q (square matrix), later expand
        one dim  (for glimpse part) and finally expanda dim and be permuted to
        [batch,hidden_dim, 1]


        vector_u:
        In this secction ref_dot and q_dor are add
            ref_dot [batch,hidden_dim, steps]
            +
            q_dot   [batch,hidden_dim,  1   ],

        later to this is applied a tanh, result is matmul by vector_vg shape [hidden_dim]
        (change by operation matmul to [1,hidden_dim]), this operation reduce the exit to
        [batch,steps]
        """

        q_g = torch.squeeze(q_g, 0)

        # Glmpse function
        #   Attention mechanism for glimpse
        # ref_g [steps, batch, hidden_dim] to [batch, hidden_sim, steps]
        ref_dot = torch.matmul(self.Wref_g, ref_g.permute(1, 2, 0))

        q_dot = torch.unsqueeze(torch.matmul(self.Wq_g, q_g.permute(1, 0)), 2).permute(
            1, 0, 2
        )
        vector_u = torch.matmul(self.vector_vg, torch.tanh(ref_dot + q_dot))
        # vector_u = self.C*torch.tanh(vector_u) # borrar si es encesario

        #   Mask values to not repeat values
        mask_union = self.mode_mask[self.mode](
            step=step,
            data_object=data_object,
            main_mask=self.mask,
            main_selection=self.selections,
            steps_qnt=self.steps_qnt,
            device=self.device,
        ).to(self.device)

        vector_u.masked_fill_(mask_union.type(torch.bool).detach(), float("-inf"))

        vector_u = F.softmax(vector_u, dim=1)

        #   ponderate values
        # r_i * p_i
        glimpse_function = ref_g.permute(1, 0, 2) * torch.unsqueeze(vector_u, 2)
        g_l = torch.sum(glimpse_function, dim=1)  # new query tensor

        # Attention mechanism for pointer (similar to glimse but use a g_l like new query tensor)
        # ref_g [batch, steps, hidden_dim] to [batch, hidden_sim, steps]
        ref_dot_pointer = torch.matmul(self.Wref, ref_g.permute(1, 2, 0))
        q_dot_pointer = torch.unsqueeze(
            torch.matmul(self.Wq, g_l.permute(1, 0)), 2
        ).permute(1, 0, 2)

        vector_u_pointer = torch.matmul(
            self.vector_v, torch.tanh(ref_dot_pointer + q_dot_pointer)
        )

        # vector_u_pointer  = self.C*torch.tanh(vector_u_pointer)

        # Valores  para los valore de inferencia
        if self.inference_mode:
            vector_u_pointer = vector_u_pointer / self.temperature

        # Entropy control with C-logits (logit-clipping) page 2 RL training
        # print("\n Antes de aplicar el control de entropia pero si aplica el valor de temperatura\n", vector_u_pointer[0])
        # vector_u_pointer = self.C*torch.tanh(vector_u_pointer) # se saturna las salidas
        # print("\n vector despues de aplicar el control de entropia",vector_u_pointer[0])

        #   Mask to not repeat values
        vector_u_pointer.masked_fill_(
            mask_union.type(torch.bool).detach(), float("-inf")
        )
        self.log.info(
            "Vector_u_pointer before to apply to mask \n %s \n", str(vector_u_pointer)
        )
        vector_u_pointer = F.softmax(vector_u_pointer, dim=1)
        self.log.info(
            "Vector of probabilities by step by step: \n%s \n\n\n",
            str(vector_u_pointer),
        )

        return vector_u_pointer

    def first_entry(
        self, batch_size, data_object: DataRepresentation, 
    ) -> torch.Tensor:  # warning cambie el valor aque era sel_nn
        r"""
        For the first element in the batch, in TSP (travelling salesman problem)
        usually uses any city because it's  interpreted as a circular permutation,
        but in this problem is not the case, for that reason, it's used a synthetic
        first element to begin with the permuation.

        Taking adventage  in  config.mode == 'n'(and config.mode = k),  we know
        previously the qnt of elements to choose and using that information for
        generating the first element. In the case of  config.mode == 'k_n' is
        different, and only is used a synthetic element without additional information.

        This can be changed to another kind entry like the entry of another network
        where the entrace is all the secuence. In this case with this entry works fine
        but with bigger secuences this can be a game changer

        **Note**: The first entry is hard coded and can be changed to be adapted to
        the new batch shape. Specially talking in the 2nd dimention shape.
        """
        # Vector [1.1, 1.4] is totally  by convenience
        first_entry = torch.tensor(
            [1.1, 1.4], dtype=torch.float32, device=self.device
        ).repeat(batch_size, 1)
        # dims (batch_len, 2)

        # if self.mode == 'n':
        #     key = 'restricted_n'
        # elif self.mode == 'k':
        #     key = 'restricted_k'

        # self.log.info('dimenciones de los batch, first entry shape%s , restriction_data %s', str(first_entry.shape), str(kwargs[key]))

        if self.mode == "n" or self.mode == "k":

            if self.mode == "n":
                restriction_data = data_object.restricted_n
            elif self.mode == "k":
                restriction_data = data_object.restricted_k

            first_entry[:, 1] = 1 / restriction_data

        first_entry = self.linear_first_entry(
            first_entry.type(torch.float32)
        ).unsqueeze(dim=0)
        first_entry = torch.tanh(first_entry)  # cuidado
        return first_entry
