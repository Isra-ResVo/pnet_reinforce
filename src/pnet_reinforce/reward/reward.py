import logging
from abc import ABC, abstractmethod
from reward.base import RewardConfig
from reward.base import BaseReward

import torch

from reward.error_func import error_function, pr_error_bound


class Reward(BaseReward):
    # this can be changed by any of the formulas given in reward.error_func module
    def __init__(self, reward_config: RewardConfig):
        super(Reward, self).__init__(reward_config=reward_config)

    def main(self, kwargs) -> torch.Tensor:

        reward = {}
        epsilon = 1e-35

        # prError for model selections
        size_subsets = (
            self.n_inferred - self.k_inferred
        ) + 1  #  subset size to have to fail before +2
        pr_error = self.error_formula(
            batch=kwargs["batch"], subsets=size_subsets, only_clouds=self.selected_clouds
        )
        # reward['prError'] = pr_error
        reward["prError"] = pr_error

        # min and max for normalization process

        maximum, minimum = self.__min_max_error(kwargs=kwargs)

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

        maximum, minimum = self.__min_max_redundancy(
            len_elements=kwargs["len_elements"]
        )
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

    def __min_max_redundancy(self, len_elements):

        r"""

        To caluclate min and max values for normalization
        Because only we have 11 clouds at this moment im working with a dict, where i can't suppose
        the redundancy beyond of this elements at this moment
        """
        mode = self.config.mode

        # return a tensor
        if mode == "n":
            maximum, minimum = [], []

            for n in self.n_inferred:
                keys = [str(i) + str(n.item()) for i in range(2, n + 1)]
                inicialVal = self.redundancy_values[keys[0]]
                maxV, minV = inicialVal, inicialVal

                for k in keys:
                    if maxV < self.redundancy_values[k]:
                        maxV = self.redundancy_values[k]

                    if minV > self.redundancy_values[k]:
                        minV = self.redundancy_values[k]

                maximum.append(maxV)
                minimum.append(minV)

            maximum = torch.tensor(maximum, dtype=torch.float32, device=self.device)
            minimum = torch.tensor(minimum, dtype=torch.float32, device=self.device)

        # return onluy one number
        elif mode == "k_n":

            # limit_keys =[]
            # for n_element_batch in len_elements:
            #     limit_keys.append( ((n_element_batch)*(n_element_batch-1)) /2 )

            # for i, key in enumerate(self.redundancy_values):
            #     val = self.redundancy_values[key]

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
                maximum.append(self.redundancy_values[key])
            maximum = torch.tensor(maximum, dtype=torch.float32, device=self.device)

        elif mode == "k":

            maximum, minimum = [], []
            for k, n in zip(self.k_inferred, len_elements):
                keys = [str(k.item()) + str(i) for i in range(k, n + 1)]
                val = self.redundancy_values[keys[0]]
                maxV, minV = val, val

                for key in keys:
                    val = self.redundancy_values[key]
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

    def __min_max_error(self, kwargs) -> torch.Tensor:

        r"""
        To calculate in the case when n y k is choosen for the system
        we use all the clouds for this propurse, this seem the qnt that get the min
        with k = 2, and max with k = n.

        Important subsets = n-k+1

        """
        batch = kwargs["batch"]
        siz, par, qnt = batch.shape
        # warning cambiar n_inferred a algo como variable de apoyo, ya que bien puede tomar los valores de k o n
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
        for (k_i, n_i) in zip(self.k_inferred, self.n_inferred):

            key = str(k_i.cpu().numpy()) + str(n_i.cpu().numpy())
            rewardR.append(
                torch.tensor(
                    self.redundancy_values.get(key, None),
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        rewardR = torch.stack(rewardR, dim=0)

        return rewardR

    def extraction_time(self):

        rwd_time = []

        for (k_i, n_i) in zip(self.k_inferred, self.n_inferred):

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
            iterable = self.redundancy_values

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
            val = self.redundancy_values[key]
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
