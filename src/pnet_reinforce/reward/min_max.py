import torch
from reward.base import BaseReward, RewardConfig

class MaxMinRedundancy(BaseReward):
    def __init__(self, reward_config: RewardConfig):
        super().__init__(reward_config)

    def max_min_redundancy(self, len_elements):

        r"""

        To caluclate min and max values for normalization because only we
        have 11 clouds at this moment I'm working with a dict, where I can't
        suppose the redundancy beyond this limit.

        Is necessary to normalize all the variables involved because some magnitudes
        can overpass any other and be the preponderant in the training avoiding
        improve the other ones.

        The only caveat behind this is that is necessary to have know the upper and
        lower limits or atleast try to get good bound.
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


class MaxMinError(BaseReward):
    def __init__(self, reward_config: RewardConfig):
        super().__init__(reward_config)

    def __indices_variable_length_and_restriction(
        self, restriction, len_elements, batch, largest
    ):
        indices = []
        for n, len_n_instance, batch_ele in zip(restriction, len_elements, batch):
            indices.append(
                torch.topk(batch_ele[0, :len_n_instance], k=n, largest=largest)
            )
        return indices

    def min_max_error(self, kwargs) -> torch.Tensor:

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
                indices = self.__indices_variable_length_and_restriction(
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
                indices = self.__indices_variable_length_and_restriction(
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
                indices = self.__indices_variable_length_and_restriction(
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
                indices = self.__indices_variable_length_and_restriction(
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
