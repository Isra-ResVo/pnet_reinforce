import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import logging


class Pointer (nn.Module):
    def __init__ (self, config,device):
        super(Pointer,self).__init__()
        self.device = device
        self.config = config

        # Mask modes according with model (mono objetive or multi objetive)
        self.mode = config.mode
        self.mode_mask = {'k_n': self.pointer_mask, 'n': self.mask_n, 'k': self.mask_k}
        self.log = logging.getLogger(__name__)
        #---------------Variables----------------------------

        #Data Input Config
        self.batch_size = config.batch_size       #default 128
        self.max_size = config.max_length         #default 7

        #Network Config
        self.hidden_dim = config.hidden_dim       #default 128 (num_neurons)
        self.initializer = torch.nn.init.xavier_normal_

        self.inference_mode = not config.train_mode
        self.C = config.C
        self.temperature =  config.temperature

        # Vector para las confuguraciones
        self.linear_first_entry = torch.nn.Linear(in_features= 2, out_features= self.hidden_dim*2, bias = True)

        # Decode layer
        self.lstm_decoder = nn.LSTM(input_size = self.hidden_dim*2, hidden_size = self.hidden_dim, num_layers = 1, batch_first = False, bidirectional = True)

        # Attention  mechanism Weights for glimpse part (def pointer mechanism)
        self.Wref_g = nn.Parameter(torch.empty(size = (self.hidden_dim * 2, self.hidden_dim * 2), dtype = torch.float32))
        nn.init.xavier_normal_(self.Wref_g)

        self.Wq_g = nn.Parameter(torch.empty(size = (self.hidden_dim*2 , self.hidden_dim*2), dtype = torch.float32))
        nn.init.xavier_normal_(self.Wq_g)


        self.vector_vg = nn.Parameter(torch.rand(size =(self.hidden_dim*2,), dtype = torch.float32))



        # Attention mechanism Weights  for Pointer Mechanism

        self.Wref = nn.Parameter(torch.empty(size = (self.hidden_dim*2, self.hidden_dim*2), dtype = torch.float32))
        nn.init.xavier_normal_(self.Wref)

        self.Wq = nn.Parameter(torch.empty(size = (self.hidden_dim*2, self.hidden_dim*2), dtype = torch.float32))
        nn.init.xavier_normal_(self.Wq)

        self.vector_v = nn.Parameter(torch.rand(size =(self.hidden_dim*2,), dtype = torch.float32))
        


    def forward (self,encoder_output, encoder_states, kwargs):

        r'''
        Encoder_output: [steps, batch, hidden_dim*directions]
        Encoder_states:
            h_state:[num_layers * num_directions, batch, hidden_size]
            s_tate: [num_layers * num_directions, batch, hidden_size]

        This two values are in array with first dimention equal 2

        '''
        self.encoder_output = encoder_output   # This will we the ref matrix in the pointer mechanism
        self.selections = []                   # Se alamacenan los resultados de las iteraciones
        self.log_probs  = []                   # here will be saved the logprobabilities of every selection


        # Dimentions of input tensor
        self.steps_qnt = encoder_output.shape[0]  # Cantidad de elementos a apuntar
        self.batch_size = encoder_output.shape[1] # Cantidad de elementos en el batch

        # Decoder context (it's internal state of coder)
        dec_s = encoder_states  # h_state:[num_layers * num_directions, batch, hidden_size]

        # First input in the decoder phase
        dec_ipt = self.first_entry(self.batch_size, kwargs).detach()

        # Masking tensor for avoid  previous selected  elements
        self.mask = torch.zeros(size=(self.batch_size, self.steps_qnt),  device = self.device)
        if self.config.variable_length:
            for i, len_element in enumerate(kwargs['len_elements']):
                if self.config.mode =='k':
                    self.mask[i,len_element:-1] = 1 # elements extra(padding) are set to 1 for be non selectionables
                else:
                    self.mask[i,len_element:-2] = 1
        
        if self.mode == 'k_n' or self.mode == 'n':
            # In the coding elements, there are 2 extra elements and 'k' is subject to at least 2 
            totalIter = (self.steps_qnt-2) * 2 - 2
        elif self.mode == 'k':
            # Here don't worry to select k elements only for original elements in the batch,
            # only exclude the stop element and that all
            totalIter = (self.steps_qnt-1)
        else: raise ValueError ('invalid value in config.mode')

        #  Ciclo de decodificación (se descuentan dos ya que k esta sujeta valer al menos 2)
        for step in range(totalIter):
            dec_ipt, dec_s = self.decode_loop(dec_ipt, dec_s, step, kwargs)

        # Secuences ordered in files [batch,secuence]
        self.selections = torch.stack(self.selections, dim = 1) # [batch,steps_secuence]

        # secuence de log sum over the steps
        self.log_probs = torch.stack(self.log_probs)   # [steps,batch]
        self.log_probs = torch.sum(self.log_probs, dim = 0) # [batch] steps dim is reduce
        self.log.info("Selections: \n%s", str(self.selections))
        self.log.info("Log probs: \n%s", str(self.log_probs))


        return self.selections, self.log_probs  # lo demás luego lo verifico


    def first_entry (self = None , batch_size = None, kwargs = None ): # warning cambie el valor aque era sel_nn
        r'''
        For the first element in the batch, in TCP usually uses a city because, it's  interpreted as a circular permutation,
        but in this problem is not the case, for that reason, it's used a synthetic first element.

        Taking adventage  in  config.mode == 'n'(and config.mode = k),  we know previously the qnt of elements to choose 
        and using that information for generating the first element. In the case of  config.mode == 'k_n' is different, and 
        only is used a synthetic element without additional information.

        '''
        # Vector [1.1, 1.4] is totally  by convenience
        first_entry = torch.tensor([1.1,1.4], dtype = torch.float32, device = self.device).repeat(batch_size,1)# dims (batch_len, 2)
        # if self.mode == 'n':
        #     key = 'restricted_n'
        # elif self.mode == 'k':
        #     key = 'restricted_k'
        
        # self.log.info('dimenciones de los batch, first entry shape%s , restriction_data %s', str(first_entry.shape), str(kwargs[key]))

        if self.mode == 'n' or self.mode == 'k':
            
            if self.mode == 'n':
                restriction_data = kwargs['restricted_n']
            elif self.mode == 'k':
                restriction_data = kwargs['restricted_k']
                
            first_entry[:,1] =  1 / restriction_data

        first_entry = self.linear_first_entry(first_entry.type(torch.float32)).unsqueeze(dim=0)
        fisrt_entry = torch.tanh(first_entry)
        return first_entry


    def decode_loop(self, decoder_loop_input, decoder_loop_state, step, kwargs):


        # Run the cell on a combination of the previous input and state
        output, state = self.lstm_decoder(decoder_loop_input, decoder_loop_state)

        # Pointer mechanism to genrate the probs of every selection
        vector_pointer = self.pointer_mechanism(self.encoder_output, output, step, kwargs) # sel_n in forward


        # Multinomial distribution
        #   Categocal distribution helps to explore different solutions over time
        #   arg.max es more aggressive without exploration, this could lead in poor solutions
        distribution_toNext_selection = Categorical(vector_pointer)

        if self.config.selection == 'categorical':

            next_selection = distribution_toNext_selection.sample()
            next_selection = next_selection.type(torch.int64)

        elif self.config.selection == 'argmax':
            next_selection = torch.argmax(vector_pointer, dim = 1)

        else:
            raise ValueError('config.selection value no valid')
        
        self.selections.append(next_selection)


        #log_pron for backprob defined for REINFORCE
    
        self.log_probs.append(distribution_toNext_selection.log_prob(next_selection))


        #update current selection and mask for funtion attention
        self.current_selection = F.one_hot(next_selection, self.steps_qnt).to(self.device)
        self.mask = self.mask.to(self.device) + self.current_selection  # la k y la stop tendran muchas sumas


        tensor_len_batch = torch.tensor(range(self.batch_size),dtype = torch.int64)

        new_decoder_input = self.encoder_output[next_selection,tensor_len_batch]     # revisar las variables
        new_decoder_input = torch.unsqueeze(new_decoder_input, dim = 0)


        return new_decoder_input, state



    def pointer_mechanism(self, ref_g, q_g, step, kwargs):
        r'''
        Wref_g,W_q \in R^d*d  and u_vector = ref \in R^d

        ref_dot:
        At the begining de ref_q must have [steps, batch, hidden_dim] later is permuted to
        size [batch,hidden_dim,steps] then multiplied by Wref_g finally has a shape
        [batch,hidden_dim, steps]

        q_dot:
        This q_g must to has a dimention [batch, hidden_dim] later is permuted to [hidden_dim, batch],
        then mutiplied by W_q (square matrix), later expand one dim  (for glimpse part) and finally expand
        a dim and be permuted to [batch,hidden_dim, 1]


        vector_u:
        In this secction ref_dot and q_dor are add
            ref_dot [batch,hidden_dim, steps]
            +
            q_dot   [batch,hidden_dim,  1   ],

        later to this is applied a tanh, result is matmul by vector_vg shape [hidden_dim] (change by operation
        matmul to [1,hidden_dim]), this operation reduce the exit to [batch,steps]
        '''
        #print("### Número de paso{}\n".format(step+1))

        self.log.info('Step: \n%s', str(step) )

        q_g = torch.squeeze(q_g,0)

        self.log.info('Tensor value ref_g \n %s \n', ref_g)

        # Glmpse function
        #   Attention mechanism for glimpse 
        ref_dot = torch.matmul(self.Wref_g, ref_g.permute(1,2,0)) # ref_g [steps, batch, hidden_dim] to [batch, hidden_sim, steps]
        self.log.info('tensor self.Wref_g this a learnable parameter: \n %s \n', str(self.Wref_g) )
        self.log.info('tensor ref_dot: \n%s\n', str(ref_dot))
        q_dot = torch.unsqueeze( torch.matmul( self.Wq_g, q_g.permute(1,0) ), 2 ).permute(1,0,2)
        vector_u =torch.matmul(self.vector_vg, torch.tanh(ref_dot + q_dot))
        # vector_u = self.C*torch.tanh(vector_u) # borrar si es encesario


        self.log.info('Vector_u before to apply mask_union \n %s \n',str(vector_u))


        #   Mask values to not repeat values

        mask_union = self.mode_mask[self.mode](step = step, kwargs = kwargs).to(self.device)
        self.log.info('mask  in evaluation: \n%s', str(mask_union))
        
        vector_u.masked_fill_(mask_union.type(torch.bool).detach(), float('-inf'))

        self.log.info('Vector u to ponderate ref tensor before to sorfmax \n %s \n', str(vector_u))
       
        vector_u = F.softmax(vector_u, dim =1)
        self.log.info('vector u to ponderate ref tensor (glimpse funtion) \n %s\n', str(vector_u))

        #   ponderate values
        glimpse_function = ref_g.permute(1,0,2) * torch.unsqueeze(vector_u,2)  #r_i * p_i
        g_l = torch.sum(glimpse_function, dim = 1) # new query tensor

        # Attention mechanism for pointer (similar to glimse but use a g_l like new query tensor)
        ref_dot_pointer = torch.matmul(self.Wref, ref_g.permute(1,2,0)) # ref_g [batch, steps, hidden_dim] to [batch, hidden_sim, steps]
        q_dot_pointer = torch.unsqueeze(torch.matmul(self.Wq, g_l.permute(1,0)),2).permute(1,0,2)

        self.log.info('Values before to matmul with self.vector_v \n %s \n', str(torch.tanh( ref_dot_pointer + q_dot_pointer)))

        vector_u_pointer =torch.matmul( self.vector_v, torch.tanh( ref_dot_pointer + q_dot_pointer ) )

        # vector_u_pointer  = self.C*torch.tanh(vector_u_pointer)

        # Valores  para los valore de inferencia
        if self.inference_mode:
            vector_u_pointer = vector_u_pointer/self.temperature

        # Entropy control with C-logits (logit-clipping) page 2 RL training
        # print("\n Antes de aplicar el control de entropia pero si aplica el valor de temperatura\n", vector_u_pointer[0])
        # vector_u_pointer = self.C*torch.tanh(vector_u_pointer) # se saturna las salidas
        # print("\n vector despues de aplicar el control de entropia",vector_u_pointer[0])

        #   Mask to not repeat values
        vector_u_pointer.masked_fill_(mask_union.type(torch.bool).detach(), float('-inf'))
        self.log.info('Vector_u_pointer before to apply to mask \n %s \n', str(vector_u_pointer))
        vector_u_pointer = F.softmax(vector_u_pointer, dim =1)
        self.log.info('Vector of probabilities by step by step: \n%s \n\n\n', str(vector_u_pointer ))

        return vector_u_pointer



    def pointer_mask(self,step,kwargs):
    
        # El valor de true en las máscara solo detenrmina que ese  valor será enmascarado
        cloud_number = -2 # Deja fuera posiciones (k_position, stop_position)
        k_position = -2
        stop_idx = -1           

        # las primera dos iteciones solo esta abilitadad la seleccion de nubes
        if  step < 2:
            mask_clouds = torch.zeros_like(self.mask)
            mask_clouds[:,cloud_number:] = 1
            mask_condition = mask_clouds + self.mask
        #se bloquea solo la k porque no puede haber una k>n
        elif step < 3:
            mask_clouds = torch.zeros_like(self.mask)
            mask_clouds[:,cloud_number] = 1  # bloquea la k
            mask_condition = mask_clouds + self.mask

        else:
            # Todo este bloque actua como un condicional, primero bloquea k sí está
            #   llega a ser mayor que la cantidad de nubes seleccionadas (1)
            #   posteriormente al resultado anterior se llega a bloquear por completo si
            #   ha aparecido previamente el  simbolo de paro (2).


            # (1) Determina la cantidad de nubes y k, hasta i.
            # obteniendo los datos relevantes de las mascaras.
            selections = torch.stack(self.selections, dim = 1) # [batch,steps_secuenfce]
            qnt_k = torch.sum((selections == self.steps_qnt -2) , dim = 1)
            qnt_stop = torch.sum((selections == self.steps_qnt -1), dim = 1)
            qnt_clouds = step - (qnt_k + qnt_stop)

            condition_k_present = qnt_k > 0
            block_q = torch.zeros_like(self.mask).to(self.device)
            block_q[:,:k_position] = 1
            mask_condition = torch.where(condition_k_present.reshape(-1,1),block_q, self.mask)
            # print( "controlando la cantidad de k aquí solo deben de estar las nubes habilitado para k menor a n \n{}".format(mask_condition[:10]))


            # (2) Bloquea todas las selecciones y solo deja el caracter de relleno.
            #     -Para bloquear la k si esta llega a ser mayor que n.
            condition_k = qnt_k + 2 < qnt_clouds
            stop_condition = self.mask[:,stop_idx:].type(torch.bool) + ~condition_k.reshape(-1,1) #bool values
            mask_all = torch.zeros_like(self.mask).to(self.device)
            mask_all[:,:stop_idx] = 1
            mask_condition = torch.where(stop_condition, mask_all, mask_condition).type(torch.bool)

        return mask_condition

    def mask_n (self,kwargs) -> "mask tensor":
        r'''
        (1) First only let the clouds be choosing  in accordance  of n_elements (this is generated by the user)
        then if this condition is meet. 
        (2) k is choose  when strictly is < n,  if k == n  then only q will be open for be take or 
        if q apper  before that k == n then the secuence over and only "q" will be choosen.
        '''
        self.log.info('argumentos de entrada%s\n ', str(args))
        cloud_number = -2 # Deja fuera posiciones (k_position, stop_position)
        stop_idx = -1
        if step <= 1:
            mask_kn = torch.zeros_like(self.mask).to(self.device)
            mask_kn[:,cloud_number:] = 1

            mask_condition = self.mask + mask_kn
       

        else:
            
             # gettting states of self.mask
            selections = torch.stack(self.selections, dim = 1) # [batch,steps_secuenfce]
            
            qnt_k = torch.sum((selections == self.steps_qnt -2) , dim = 1)
            qnt_stop = torch.sum((selections == self.steps_qnt -1), dim = 1)
            qnt_clouds = step - (qnt_k + qnt_stop)
            self.log.info('cantidad de nubes qnt_clouds \n %s', qnt_clouds)
            
            # (1) Only selection of clouds
            cloud_cond = (qnt_clouds >= kwargs['restricted_n']).reshape(-1,1)
            mask_clouds = torch.zeros_like(self.mask, dtype = torch.int64).to(self.device)
            mask_clouds[:,:cloud_number] = 1

            mask_extra = torch.zeros_like(self.mask, dtype =torch.int64).to(self.device)
            mask_extra[:,cloud_number:] = 1 # ????? revisar este

            mask_clouds = torch.where(cloud_cond, mask_clouds, mask_extra)
            mask_withOutK = self.mask
            mask_withOutK[:,-2] = 0
            mask_condition = mask_clouds + mask_withOutK

            # (2) k selection with k <= n and q condition
            k_condition = (qnt_k+2) >= kwargs['restricted_n']
            self.log.info('self.mask: %s\n', str(self.mask) )
            self.log.info('k condition %s\n', str(k_condition))

            q_condition = self.mask[:,stop_idx].type(torch.bool)
            self.log.info('q_condition %s\n', str(q_condition))
            condition_qk = (k_condition + q_condition).reshape(-1,1) # k or q (bool)

            self.log.info('condition_qk %s\n\n\n', str(condition_qk))

            mask_qk = torch.ones_like(self.mask).to(self.device)
            mask_qk[:,stop_idx] = 0 # only stop condition available

            mask_condition = torch.where(condition_qk, mask_qk, mask_condition)

        return mask_condition.type(torch.bool)


    def mask_k(self, kwargs):
        # This function permits to have minimum k clouds and C clouds (all the batch elements)
        
        stop_idx = -1
        # Validation k elements greater or equal to 2
        allElementsGreaterEqual2 = torch.any(kwargs['restricted_k'] >= 2)
        if not allElementsGreaterEqual2: raise ValueError('invalid k paramer')

        self.log.info('\n\n\n')
        self.log.info('self.mask value %s \n', str(self.mask) )

        if  step <= 1: # Take minimum 2 elements in the batch
            mask_stopElement = torch.zeros_like(self.mask, device = self.device)
            mask_stopElement[:,stop_idx] = 1

            mask_condition = self.mask + mask_stopElement

        else:
            mask_stopElement = torch.zeros_like(self.mask, device = self.device)
            mask_stopElement[:,stop_idx] = 1
            maskPlusBlockStop = self.mask + mask_stopElement


            stopByItemsSelected =  step >= kwargs['restricted_k'] # This allows to grab items with minimum requirement of k items previosly selected.

            self.log.info('stopByItemSelected %s\n', str( stopByItemsSelected))
            # mask that block  the stopElement  if  n < k
            mask_condition = torch.where(stopByItemsSelected.reshape(-1, 1), self.mask, maskPlusBlockStop) # (condition, if true, else)
        
            conditionStopElement = self.mask[:,stop_idx].type(torch.bool).reshape(-1,1)
            self.log.info('conditionStopElement %s\n', str(conditionStopElement))
            mask_clouds = torch.ones_like(self.mask, device = self.device)
            mask_clouds[:,-1] = 0

            mask_condition = torch.where(conditionStopElement, mask_clouds, mask_condition)

        return mask_condition.type(torch.bool)

   