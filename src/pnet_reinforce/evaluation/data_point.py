from abc import ABC

import torch

from reward.reward_representation import DataSolution
from generator.data_interface.data import DataRepresentation
from reward.error_func import pr_error_bound

selected_error_formula = pr_error_bound


class Point(object):
    """
    This object is for the representation of one point of information to
    graph and make comparatin with other points of data.

    Remember the probabilities are in log. Becuase the probabilities too small
    overflow the float in the calcualations.
    """

    def __init__(
        self,
        reward_grouped: DataSolution,
        data_object: DataRepresentation,
        index: int,
    ):
        # Config data
        self.config = reward_grouped.config

        # Essential data
        # remember the probability of error is log.
        self.probability_of_error = reward_grouped.probability_of_error[index]
        self.normalized_redundancy = reward_grouped.normalized_redundancy[index]
        self.redundancy = reward_grouped.redundancy[index]
        self.n_inferred = reward_grouped.n_inferred[index]
        self.k_inferred = reward_grouped.k_inferred[index]
        self.selected_clouds = reward_grouped.selected_clouds[index]
        self.elements_length = data_object.elements_length[index]

        # Probability of error referential  vals
        #   This are provided with PointRefereces tool.
        self.probability_of_error_no_log = None  # "pr_ln"
        self.probability_of_error_normalized = None  # "prn_ln"
        self.probability_of_error_no_log_normalized = None  # "prn"
        self.maximum_value_in_batch_element = None
        self.minimum_value_in_batch_element = None
        self.degradation_wrt_minimum = None

        #   Multiple points
        self.all_element_batch_error_probabilities = None
        self.all_element_batch_error_probabilities_normalized = None
        self.all_element_batch_error_probabilities_no_log = None

        # Redundancy values referencial values

        self.redundancy_all_values_of_element_batch_normalized = None
        self.redundancy_all_values_of_element_batch = None
        self.redundancy_degradation_wrt_minimum = None
        self.redundancy_minimum = None
        self.redundancy_maximum = None

        # weighted point
        self.weighted_objective = None
        self.all_element_batch_ponderate_objective = None
        self.all_element_batch_ponderate_objective_normalized = None


class PointReferences(ABC):
    def __init__(self, point_object: Point):
        self.point_object = point_object