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



def error_function(
    self=None, batch=None, subsets: "list" = None, only_clouds: "list" = None
) -> "float":
    # Note this function uses itertools.combinations and for this reason could be too computationally expensive

    r"""
    Here the variables subsets and only clouds are dtype list
    subsets: contains list for selec j items (j = n-k+1 for every batch element)
        Example:
            [2,3,4]
    only_clouds: contain a list with index of elements, to be drag from batch values.
            [[0,3,4,5],
            [0,3,4,5],
            [0,3,4,5]]

    Note: very inefficient way to calculate this is better to take the values and make a bound
    because the complexity is too high for this porpouse
    """

    pr_error = []
    for idx_batch, (subset, clouds) in enumerate(zip(subsets, only_clouds)):

        clouds = torch.sort(clouds)[
            0
        ]  # This is necesary to get the same result (float point operations)
        combinations = itertools.combinations(clouds, subset)  # -> iter_object
        combinations = torch.stack(
            [torch.stack(comb) for comb in combinations]
        )  # -> Tensor

        buffer_pr_error = 0

        for combination in combinations:

            A_valuesT = batch[idx_batch, 0, combination]
            A_valores = torch.prod(A_valuesT)

            # Getting the complement values for combinations.
            mask = ~torch.sum(clouds == combination.reshape(-1, 1), dim=0).type(
                torch.bool
            )
            complement_A = clouds[mask]
            A_comp_values = torch.prod((1 - batch[idx_batch, 0, complement_A]))
            Pr_sub = A_valores * A_comp_values
            buffer_pr_error += Pr_sub

        pr_error.append(buffer_pr_error)
    pr_error = torch.stack(pr_error)
    pr_error = torch.log(pr_error)  # to avoid change a lot of code

    return pr_error


def pr_error_bound(
    self=None, batch=None, subsets: list = None, only_clouds: list = None
) -> torch.Tensor:

    r"""
    This function calculate the probability of failure by taking the combinations of
    elected clouds quantity by the model.

    This bound greater (>) than original formula, and is almost O(n) in complexity,
    converserly the original it is almost O(n^k).

    The main idea is only to use binomial coefficient to get all the possible combinations and
    multiply it by the worst cloud stats among the all them.

    """

    def factorial2(tensor):
        return (tensor + 1).to(torch.float32).lgamma().exp()

    def log_binomial_coefficient(n, k):
        first_exp = factorial2(n)
        second_exp = factorial2(k) * factorial2(n - k)
        return torch.log(first_exp / second_exp)

    # Factorial value for every cloud
    number_of_clouds = []
    for element in only_clouds:
        number_of_clouds.append(len(element))

    number_of_clouds = torch.tensor(number_of_clouds)
    if subsets.is_cuda:
        number_of_clouds = number_of_clouds.cuda()

    logBinomialCoefficient = log_binomial_coefficient(n=number_of_clouds, k=subsets)

    # calculating the error bound of the clouds based on logBinomialCoefficient
    prError = []
    for i, (subset, clouds, binomial) in enumerate(
        zip(subsets, only_clouds, logBinomialCoefficient)
    ):
        clouds = torch.sort(clouds)[
            0
        ]  # when you work with high precition float, to make operation in different order can give similar but diffetent values

        combination = torch.topk(batch[i, 0, clouds], subset, largest=True)[1]
        combination = clouds[combination]

        sumErrorCombination = torch.sum(torch.log(batch[i, 0, combination]))

        maskComplement = torch.tensor(
            [cloud not in combination for cloud in clouds], dtype=torch.bool
        )
        complementIndex = clouds[maskComplement]

        sumErrorComplement = torch.sum(torch.log(1 - batch[i, 0, complementIndex]))

        prError.append(sumErrorCombination + sumErrorComplement + binomial)

    return torch.stack(prError)


selected_error_formula = pr_error_bound


class Reward(object):

    error_formula = selected_error_formula

    def __init__(self, selections, device, qnt_steps, config, value_k=None):
        self.k_key = qnt_steps
        self.log = logging.getLogger(__name__)
        self.config = config
        self.device = device
        self.arrangenment = selections
        if self.config.mode == "k":
            if value_k is None:
                raise ValueError(" value_k must be given")
            self.only_clouds, self.k_sum, self.sel_n = self.get_k_and_clouds(value_k)
        else:
            self.only_clouds, self.k_sum, self.sel_n = self.get_k_and_clouds()

    def get_k_and_clouds(self, value_k=None):

        if self.config.mode == "k":
            # validadation
            if value_k is None:
                raise ValueError("value_k is None, k must be provided")
            k_sum = value_k
            # Element in the batch that serves as either stop condition or like fill  element
            #  In this case only reassingned a variable name for readability
            stopElement_mask = ~(self.arrangenment == self.k_key)

            cloud_list = []
            for mask, arrange in zip(stopElement_mask, self.arrangenment):
                cloud_list.append(torch.masked_select(arrange, mask))

            qnt_clouds = torch.sum(stopElement_mask, dim=1)

        else:
            # This section serves for n and n_k configurations

            k_qnt = self.arrangenment == self.k_key
            k_sum = torch.sum(k_qnt.type(torch.int64), dim=1) + 2

            stop_position = self.arrangenment == self.k_key + 1
            mask_clouds = ~(k_qnt + stop_position).type(torch.bool)

            cloud_list = []
            for mask, arrange in zip(mask_clouds, self.arrangenment):
                cloud_list.append(torch.masked_select(arrange, mask))

            qnt_clouds = torch.sum(mask_clouds, dim=1)

        return cloud_list, k_sum, qnt_clouds

    def main(self, kwargs) -> torch.Tensor:

        reward = {}
        epsilon = 1e-35

        # prError for model selections
        size_subsets = (
            self.sel_n - self.k_sum
        ) + 1  #  subset size to have to fail before +2
        pr_error = self.error_formula(
            batch=kwargs["batch"], subsets=size_subsets, only_clouds=self.only_clouds
        )
        # reward['prError'] = pr_error
        reward["prError"] = pr_error

        # min and max for normalization process

        maximum, minimum = self.min_max_error(kwargs=kwargs)

        print("probabilidad de error", pr_error)
        print("maximum", maximum)
        print("minimum", minimum)

        # Log probabilities
        # if self.config.log:
        #     logError = torch.log(pr_error)
        #     maximum = torch.log(maximum)
        #     minimum = torch.log(minimum)
        #     reward['normError'] = (logError - minimum)/(maximum - minimum + epsilon)
        # else:
        #     normError = (pr_error - minimum) / (maximum - minimum +epsilon)
        #     reward['normError'] = normError

        normError = (pr_error - minimum) / (maximum - minimum + epsilon)
        reward["normError"] = normError

        print("reward[normError]", reward["normError"])
        # print('valores normalizados de error de fallo', reward['normError'])

        # Redundancy and normalization
        reward["redundancy"] = self.redundancy()
        # print('valor de redundancia', reward['redundancy'])

        maximum, minimum = self.min_max_redundancy(len_elements=kwargs["len_elements"])
        # print('minimos y maximos\n', maximum,'\n', minimum)

        reward["normRed"] = (reward["redundancy"] - minimum) / (
            maximum - minimum + epsilon
        )
        # print('valores normalizados de redundancia', reward['normRed'])

        # Ponderate
        reward["ponderate"] = (
            self.config.wo[0] * reward["normError"]
            + self.config.wo[1] * reward["normRed"]
        )
        # reward['ponderate'] = 10/(1-torch.log(reward['ponderate'])) # realizar cambios d

        return reward

    def min_max_redundancy(self, len_elements):

        r"""

        To caluclate min and max values for normalization
        Because only we have 11 clouds at this moment im working with a dict, where i can't suppose
        the redundancy beyond of this elements at this moment
        """
        mode = self.config.mode

        # return a tensor
        if mode == "n":
            maximum, minimum = [], []

            for n in self.sel_n:
                keys = [str(i) + str(n.item()) for i in range(2, n + 1)]
                inicialVal = self.dictRedundancy[keys[0]]
                maxV, minV = inicialVal, inicialVal

                for k in keys:
                    if maxV < self.dictRedundancy[k]:
                        maxV = self.dictRedundancy[k]

                    if minV > self.dictRedundancy[k]:
                        minV = self.dictRedundancy[k]

                maximum.append(maxV)
                minimum.append(minV)

            maximum = torch.tensor(maximum, dtype=torch.float32, device=self.device)
            minimum = torch.tensor(minimum, dtype=torch.float32, device=self.device)

        # return onluy one number
        elif mode == "k_n":

            # limit_keys =[]
            # for n_element_batch in len_elements:
            #     limit_keys.append( ((n_element_batch)*(n_element_batch-1)) /2 )

            # for i, key in enumerate(self.dictRedundancy):
            #     val = self.dictRedundancy[key]

            #     if i == 0:
            #         minimum = val
            #         maximum = val

            #     else:
            #         if minimum > val: minimum = val
            #         if maximum < val: maximum = val

            #     if i == limit_keys[i]-1:
            #         break

            # nueva parte
            batch_size = len(len_elements)
            minimum = (
                torch.ones(size=(batch_size,), dtype=torch.float32, device=self.device)
                * 2
            )
            keys = [str(2) + str(i) for i in len_elements]
            maximum = []
            for key in keys:
                maximum.append(self.dictRedundancy[key])
            maximum = torch.tensor(maximum, dtype=torch.float32, device=self.device)

        elif mode == "k":

            maximum, minimum = [], []
            for k, n in zip(self.k_sum, len_elements):
                keys = [str(k.item()) + str(i) for i in range(k, n + 1)]
                self.log.info("valores de k: %s", str(keys))
                val = self.dictRedundancy[keys[0]]
                maxV, minV = val, val

                for key in keys:
                    val = self.dictRedundancy[key]
                    if maxV < val:
                        maxV = val
                    if minV > val:
                        minV = val

                maximum.append(maxV)
                minimum.append(minV)

            maximum = torch.tensor(maximum, dtype=torch.float32, device=self.device)
            minimum = torch.tensor(minimum, dtype=torch.float32, device=self.device)

        else:
            raise ValueError("not valid mode, it's only accepted n, k and n_k")

        return maximum, minimum

    def min_max_error(self, kwargs) -> torch.Tensor:

        r"""
        To calculate in the case when n y k is choosen for the system
        we use all the clouds for this propurse, this seem the qnt that get the min
        with k = 2, and max with k = n.

        Important subsets = n-k+1

        """
        batch = kwargs["batch"]
        siz, par, qnt = batch.shape
        # warning cambiar sel_n a algo como variable de apoyo, ya que bien puede tomar los valores de k o n
        # segun la configuracion que se tiene.

        # This section if for obtain max value of error

        def indices_variable_length_and_restriction(
            restriction, len_elements, batch, largest
        ):
            indices = []
            for n, len_n_instance, batch_ele in zip(restriction, len_elements, batch):
                indices.append(
                    torch.topk(batch_ele[0, :len_n_instance], k=n, largest=largest)
                )
            return indices

        indices = []
        if self.config.mode == "k_n":  # warning

            if self.config.variable_length and "len_elements" in kwargs:
                for len_n_instance in kwargs["len_elements"]:
                    indices.append(
                        torch.arange(
                            0, len_n_instance, dtype=torch.int64, device=self.device
                        )
                    )
            else:
                indices = torch.arange(
                    0, qnt, dtype=torch.int64, device=self.device
                ).repeat(siz, 1)

            subsets = torch.ones(size=(siz,), dtype=torch.int64, device=self.device)

        elif self.config.mode == "n" and "restricted_n" in kwargs:
            if self.config.variable_length and "len_elements" in kwargs:
                indices = indices_variable_length_and_restriction(
                    kwargs["restricted_n"], kwargs["len_elements"], batch, True
                )

            else:
                for n, batchEle in zip(kwargs["restricted_n"], batch):
                    # Taking indices of k elements with lower values
                    indices.append(torch.topk(batchEle[0], k=n, largest=True)[1])

            subsets = torch.ones(size=(siz,), dtype=torch.int64, device=self.device)

        elif self.config.mode == "k" and "restricted_k" in kwargs:
            # In this option is special, only when k = n it's the higher value of pr_error and, the lowers when
            #   n>k we have more conbinations and consecuently a lower pr_error
            if self.config.variable_length:
                indices = indices_variable_length_and_restriction(
                    kwargs["restricted_k"], kwargs["len_elements"], batch, True
                )
            else:
                for k, batchEle in zip(kwargs["restricted_k"], batch):
                    indices.append(torch.topk(batchEle[0], k=k, largest=True)[1])

            subsets = torch.ones(size=(siz,), dtype=torch.int64, device=self.device)

        else:
            raise ValueError(
                "This value is not valid, only accept n, k and n_k configurations"
            )

        max_error = self.error_formula(
            batch=batch, subsets=subsets, only_clouds=indices
        )

        # This section is for calculate the value from min value

        if self.config.mode == "k_n":
            indices = []
            if self.config.variable_length:
                for len_n_instance in kwargs["len_elements"]:
                    indices.append(
                        torch.arange(
                            0, len_n_instance, dtype=torch.int64, device=self.device
                        )
                    )
                subsets = [i - 1 for i in kwargs["len_elements"]]
                subsets = torch.tensor(subsets, dtype=torch.int64, device=self.device)
            else:
                val = qnt
                indices = torch.arange(
                    start=0, end=qnt, dtype=torch.int64, device=self.device
                ).repeat(siz, 1)
                subsets = (
                    torch.ones(size=(siz,), dtype=torch.int64, device=self.device) * val
                ) - 1

        elif self.config.mode == "n" and "restricted_n" in kwargs:
            if self.config.variable_length and "len_elements" in kwargs:
                indices = indices_variable_length_and_restriction(
                    kwargs["restricted_n"], kwargs["len_elements"], batch, False
                )
                subsets = [i - 1 for i in kwargs["len_elements"]]

            else:
                val = kwargs["restricted_n"]
                for n, batchEle in zip(kwargs["restricted_n"], batch):
                    # Taking indices of k elements with lowest values
                    indices.append(torch.topk(batchEle[0], k=n, largest=False)[1])
                subsets = (
                    torch.ones(size=(siz,), dtype=torch.int64, device=self.device) * val
                ) - 1

        elif self.config.mode == "k" and "restricted_k" in kwargs:

            if self.config.variable_length:
                indices = indices_variable_length_and_restriction(
                    kwargs["len_elements"], kwargs["len_elements"], batch, False
                )
                subsets = [
                    i - j + 1
                    for i, j in zip(kwargs["len_elements"], kwargs["restricted_k"])
                ]
            else:
                # To get the lowest pr_error is necesarary take all clouds in the batch
                restricted_k = kwargs["restricted_k"]
                indices = torch.arange(
                    0, qnt, dtype=torch.int64, device=self.device
                ).repeat(siz, 1)
                val = qnt - restricted_k + 1
                subsets = (
                    torch.ones(size=(siz,), dtype=torch.int64, device=self.device) * val
                )
        else:
            raise ValueError("One condition is violated")

        min_error = self.error_formula(
            batch=batch, subsets=subsets, only_clouds=indices
        )

        return max_error, min_error

    def redundancy(self):
        r"""
        This values are experimental values from real comfigurations. This values were retrivied from previous
        literature

        """

        rewardR = []
        for (k_i, n_i) in zip(self.k_sum, self.sel_n):

            key = str(k_i.cpu().numpy()) + str(n_i.cpu().numpy())
            rewardR.append(
                torch.tensor(
                    self.dictRedundancy.get(key, None),
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        rewardR = torch.stack(rewardR, dim=0)

        return rewardR

    def extraction_time(self):

        cloud_n = self.sel_n
        k_qnt = self.k_sum

        cloud_n = cloud_n.to("cpu")
        k_qnt = k_qnt("cpu")

        rwd_time = []

        for (k_i, n_i) in zip(self.k_sum, self.cl):

            k = str(k_i.cpu().numpy())
            n = str(n_i.cpu().numpy())
            rwd_time.append(
                torch.tensor(self.time.get(k + n, 12345))
            )  # se castigan valores  soluciones no validas

        return torch.stack(rwd_time, dim=0)

    # dicts becuase are too large these are in the bottom

    def redundancyValsPlot(self, point, config, kwargs, index):
        # This funtion is for creating the values for plotting

        # agreagar index and kwargs
        mode = config.mode
        redundancy = []

        if mode == "k_n":
            if config.variable_length:
                n = kwargs["len_elements"][index]
            else:
                n = point["batchQntClouds"]

            limitKeys = ((n) * (n - 1)) / 2
            iterable = self.dictRedundancy

        elif mode == "n":

            n = point["n_position"]
            iterable = [str(i) + str(n.item()) for i in range(2, n + 1)]

        elif mode == "k":
            if config.variable_length:
                n = kwargs["len_elements"][index]
            else:
                n = point["batchQntClouds"]

            k = point["k_position"]
            iterable = [str(k.item()) + str(i) for i in range(k, n + 1)]

        for i, key in enumerate(iterable):
            val = self.dictRedundancy[key]
            redundancy.append(val)

            if i == 0:
                minimum = val
                maximum = val

            else:
                if minimum > val:
                    minimum = val
                if maximum < val:
                    maximum = val

            if mode == "k_n":
                if i == limitKeys - 1:
                    break

        redundancy = torch.tensor(redundancy, dtype=torch.float32)
        redundancy_original = redundancy

        logredundancy = False
        if logredundancy:

            redundancy = torch.log(redundancy)
            maximum = torch.log(torch.tensor(maximum, dtype=torch.float32))
            minimum = torch.log(torch.tensor(minimum, dtype=torch.float32))

        epsilon = 1e-35
        redundancy = (redundancy - minimum) / (maximum - minimum + epsilon)
        ratio = point["redundancy"] / minimum

        return redundancy, redundancy_original, ratio, minimum, maximum

    dictRedundancy = {
        "22": 2,
        "23": 3,
        "33": 2,
        "24": 4,
        "34": 2.67,
        "44": 2,
        "25": 5,
        "35": 3.3,
        "45": 2.5,
        "55": 2,
        "26": 6,
        "36": 4,
        "46": 3,
        "56": 2.4,
        "66": 2,
        "27": 7,
        "37": 4.67,
        "47": 3.50,
        "57": 2.80,
        "67": 2.33,
        "77": 2,
        "28": 8,
        "38": 5.33,
        "48": 4.00,
        "58": 3.20,
        "68": 2.67,
        "78": 2.29,
        "88": 2,
        "29": 9,
        "39": 6,
        "49": 4.5,
        "59": 3.60,
        "69": 3.00,
        "79": 2.57,
        "89": 2.25,
        "99": 2,
        "210": 10,
        "310": 6.67,
        "410": 5,
        "510": 4,
        "610": 3.33,
        "710": 2.86,
        "810": 2.50,
        "910": 2.22,
        "1010": 2.00,
        "211": 11,
        "311": 7.33,
        "411": 5.50,
        "511": 4.40,
        "611": 3.67,
        "711": 3.14,
        "811": 2.75,
        "911": 2.44,
        "1011": 2.20,
        "1111": 2,
    }

    time = {
        "22": 255,  # revisar
        "23": 275,  #
        "33": 267,
        "24": 354,
        "34": 296,
        "44": 269,
        "25": 472.91,
        "35": 354.24,
        "45": 326.80,
        "55": 292.93,
        "26": 567.10,
        "36": 433.62,
        "46": 368.28,
        "56": 333.78,
        "66": 293.20,
        "27": 636.15,
        "37": 485.77,
        "47": 412.74,
        "57": 374.42,
        "67": 337.15,
        "77": 306.36,
        "28": 756.47,
        "38": 552.41,
        "48": 455.53,
        "58": 394.46,
        "68": 369.37,
        "78": 323.95,
        "88": 303.01,
        "29": 806.42,
        "39": 605.95,
        "49": 493.77,
        "59": 441.51,
        "69": 409.11,
        "79": 350.38,
        "89": 328.97,
        "99": 301.11,
        "210": 838.49,
        "310": 614.82,
        "410": 520.69,
        "510": 436.35,
        "610": 396.58,
        "710": 358.53,
        "810": 345.72,
        "910": 314.36,
        "1010": 295.72,
        "211": 914.58,
        "311": 646.07,
        "411": 529.79,
        "511": 474.61,
        "611": 412.24,
        "711": 386.46,
        "811": 356.12,
        "911": 329.70,
        "1011": 301.23,
        "1111": 292.26,
    }


# Funtions declarated  to Reward class in outer scope
# Caution be aware with the self argument, bacaise is the first argument  from the system


def pr_vals_2_plot(toCompareInPlot, kwargs, point, config, device, idxEle=0):
    # agregar config
    # All this only works for one element!
    local_error_formula = selected_error_formula
    mode = config.mode
    log = config.log

    batch = kwargs["batch"]
    indixes = kwargs["indices"]
    buffer = []
    bestWorst = [False, True]

    if mode == "k_n":
        # batch_optiPessi = torch.tensor(optimisticPessimistic(indices = indixes[idxEle]))
        # for i, _ in enumerate(batch_optiPessi):
        #       buffer.append(all_combinations(batch = batch_optiPessi, idx_batch = i, device = device))
        for boolean in bestWorst:
            buffer.append(
                all_combinations(
                    batch, config, idxEle, kwargs, device=device, largest=boolean
                )
            )

    elif mode == "n":
        n = point["n_position"]
        selections = point["onlyClouds"]

        # Replicating the first element for the batch

        siz = n - 1
        batch = batch[idxEle].repeat(siz, 1, 1)

        # First elements is to generate all the possible solutions with that selections of clouds
        # Second element is to generate all the posbible solutions with the best elements in batch

        # onlyCloudsBatch = [selections.repeat(siz,1),  torch.topk(batch[idxEle][0],k = n, largest = False)[1].repeat(siz,1)]
        onlyCloudsBatch = []
        for boolean in bestWorst:
            if config.variable_length:
                len_element = kwargs["len_elements"][idxEle]
                onlyCloudsBatch.append(
                    torch.topk(batch[idxEle, 0, :len_element], k=n, largest=boolean)[
                        1
                    ].repeat(siz, 1)
                )
            else:
                onlyCloudsBatch.append(
                    torch.topk(batch[idxEle][0], k=n, largest=boolean)[1].repeat(siz, 1)
                )

        k = torch.arange(2, n + 1, dtype=torch.int64, device=device)
        subsets = n - k + 1

    elif mode == "k":

        k = point["k_position"]
        selections = point["onlyClouds"]

        if config.variable_length:
            qnt_clouds = kwargs["len_elements"][idxEle]
            len_element = kwargs["len_element"][idxEle]
        else:
            qnt_clouds = batch.shape[2]

        siz = qnt_clouds - k + 1
        batch = batch[idxEle].repeat(siz, 1, 1)
        elementsToTake = torch.arange(
            k, qnt_clouds + 1, dtype=torch.int64, device=device
        )

        onlyCloudsBatch = []
        for config_topk in bestWorst:
            onlyClouds = []
            for quantity in elementsToTake:
                if config.variable_length:
                    onlyClouds.append(
                        torch.topk(
                            batch[idxEle, 0, :len_element],
                            k=quantity,
                            largest=config_topk,
                        )[1]
                    )
                else:
                    onlyClouds.append(
                        torch.topk(batch[idxEle][0], k=quantity, largest=config_topk)[1]
                    )
            onlyCloudsBatch.append(onlyClouds)

        subsets = torch.arange(1, siz + 1, dtype=torch.int64, device=device)

    else:
        raise ValueError("invalid mode fix")

    if mode == "k" or mode == "n":
        for onlyCloud in onlyCloudsBatch:
            buffer.append(
                local_error_formula(batch=batch, subsets=subsets, only_clouds=onlyCloud)
            )

    buffer = torch.stack(buffer)
    maximum = torch.max(buffer)
    minimum = torch.min(buffer)

    # Only graph the option with best elements[0] because there are too many combinations, the other
    # it's out for tutur recomendation.
    buffer = buffer[0]

    pr_error_2graph_ln = buffer
    pr_error_2graph = torch.exp(buffer)

    degradation = point["value_error"] / minimum
    # point['prErrorOri'] = point['value_error']
    point["pr_ln"] = point["value_error"]
    point["pr_error"] = torch.exp(point["value_error"])

    epsilon = 1e-35
    buffer_normalized = (buffer - minimum) / (
        maximum - minimum - epsilon
    )  # it's in log(ln)
    point["prn_ln"] = (point["pr_ln"] - minimum) / (maximum - minimum + epsilon)
    point["prn"] = (torch.exp(point["pr_ln"]) - torch.exp(minimum)) / (
        torch.exp(maximum) - torch.exp(minimum) + epsilon
    )

    # no log values

    # point['prError'] debe de ser point['prn']
    # point['prAbs'] debe de ser point['prError']
    # point['prlog'] debe de ser point['pr_ln']

    # for element in buffer:
    #     toCompareInPlot.append(element)

    return (
        (pr_error_2graph_ln, pr_error_2graph),
        buffer_normalized,
        degradation,
        minimum,
        maximum,
    )


def all_combinations(
    batch, config, idx_batch=0, kwargs=None, device="cpu", largest=False
) -> torch.tensor:
    r"""
    All the combinations for k and n, with all the providers taken in the batch, for instance:
    if we have n = 3 in the batch,  then the conbination for tuple(n,k) are
        (2,2),(2,3),(3,3)....
    where  2 <= k  <=n

    idx_batch: is to choose the element to work with, default = 0

    Note: This only work with one element

    """
    local_error_formula = selected_error_formula
    # agregar config and kwargs
    if config.variable_length:
        len_element = kwargs["len_elements"][idx_batch]
        cloud_qnts = torch.arange(start=2, end=len_element + 1, device=device)

    else:
        dim_2 = batch.shape[2]
        cloud_qnts = torch.arange(start=2, end=dim_2 + 1, device=device)

    ks = [
        torch.arange(start=2, end=cloud_qnt + 1, device=device)
        for cloud_qnt in cloud_qnts
    ]

    # calculating the values for all the conbination of error tupole (k,n) posible
    all_values = []
    for cloud_element, ks_element in zip(cloud_qnts, ks):
        k_sum = ks_element
        sel_n = cloud_element.repeat(k_sum.shape[0])
        subsets = sel_n - k_sum + 1

        # Best k elements
        if config.variable_length:
            len_element = kwargs["len_elements"][idx_batch]
            _, index = torch.topk(
                input=batch[idx_batch, 0, :len_element],
                k=cloud_element,
                largest=largest,
            )  # probable bug

        else:
            _, index = torch.topk(
                input=batch[idx_batch][0], k=cloud_element, largest=largest
            )  # probable bug

        only_clouds = index.repeat(
            k_sum.shape[0], 1
        )  # índices de lo elementos a selecionar
        batch_cloned = (
            batch[idx_batch].clone().unsqueeze(dim=0).repeat(k_sum.shape[0], 1, 1)
        )

        # one element error probability
        pr_error = local_error_formula(
            batch=batch_cloned, subsets=subsets, only_clouds=only_clouds
        )

        all_values.append(pr_error)
    all_values = torch.cat(all_values, dim=0)

    return all_values
