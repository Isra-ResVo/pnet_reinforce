from abc import ABC, abstractmethod

import torch

from generator.generator.initial_data import TIME, REDUNDANCY
from reward.error_func import error_function, pr_error_bound


class RewardConfig(object):

    """
    This function save all the necessesary data to process the
    required reward used in traning based in the solutions generated
    by the network.
    """

    def __init__(self, selections, device, qnt_steps, config, value_k=None):
        self.wild_card_element = qnt_steps
        self.config = config
        self.device = device
        self.model_selections = selections

        self.error_formula = pr_error_bound

        self.redundancy_values = REDUNDANCY
        self.time = TIME

        # Generate values
        if self.config.mode == "k":
            if value_k is None:
                raise ValueError(" value_k must be given")
            self.selected_clouds, self.k_inferred, self.n_inferred = self.get_k_and_clouds(value_k)
        else:
            self.selected_clouds, self.k_inferred, self.n_inferred = self.get_k_and_clouds()

    def get_k_and_clouds(self, value_k=None):
        """
        The model select items of every element in the batch in a loop way. With this
        context before this begin the `pointer_network` adds an extra element/token
        as wild card.
        When k token is selected the next iterations only allow to select it repeteadly and
        the quantitie of times is selected determines the value of k.

        Maybe the wild card can be reemplace by another network but another problem
        arises. How to enforce a valid solution?

        return:
        --------
        `selected_clouds`: List of items selected by the model
        `k_inferred: array`: Value of inferred of k
        `n_inferred: array`: Value of inferred of n' 
        """
        selected_clouds = []
        if self.config.mode == "k":
            k_inferred = value_k
            # Element in the batch that serves as either stop condition or like fill  element
            #  In this case only reassingned a variable name for readability
            wild_card_mask = ~(self.model_selections == self.wild_card_element)

            for mask, arrange in zip(wild_card_mask, self.model_selections):
                selected_clouds.append(torch.masked_select(arrange, mask))

            n_inferred = torch.sum(wild_card_mask, dim=1)

        else:
            # This section serves for n and n_k configurations

            k_qnt = self.model_selections == self.wild_card_element
            k_inferred = torch.sum(k_qnt.type(torch.int64), dim=1) + 2

            stop_position = self.model_selections == self.wild_card_element + 1
            mask_clouds = ~(k_qnt + stop_position).type(torch.bool)

            for mask, arrange in zip(mask_clouds, self.model_selections):
                selected_clouds.append(torch.masked_select(arrange, mask))

            n_inferred = torch.sum(mask_clouds, dim=1)

        return selected_clouds, k_inferred, n_inferred


class BaseReward(ABC):
    """Base class to save all configuration to genrate reward
    and related data.

    Args:
        ABC (_type_): meta class
    """    
    def __init__(self, reward_config: RewardConfig):
        """

        Args:
            reward_config (RewardConfig): _description_
        """        
        self.k_inferred = reward_config.k_inferred
        self.n_inferred = reward_config.n_inferred
        self.selected_clouds = reward_config.selected_clouds
        self.device = reward_config.device

        self.redundancy_values = reward_config.redundancy_values
        self.time = reward_config.time
        self.error_formula = pr_error_bound

        self.config = reward_config.config
        self.reward_config = reward_config