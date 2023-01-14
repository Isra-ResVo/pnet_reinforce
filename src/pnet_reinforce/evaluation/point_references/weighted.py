import torch

from reward.reward_representation import DataSolution
from generator.data_interface.data import DataRepresentation
from reward.error_func import pr_error_bound

from evaluation.data_point import Point, PointReferences

class WeightedObjectiveReferences(PointReferences):
    def __init__(self, point_object: Point, config):
        super().__init__(point_object=point_object)
        self.config = config

    def add(self):
        r"""
        Adds the next values to the Point object but previously this object has to:

        self.point_object.wo
        self.weighted_objective
        self.all_element_batch_ponderate_objective
        self.all_element_batch_ponderate_objective_normalized
        """
        self.point_object.weighted_objective = (
            self.point_object.probability_of_error_normalized * self.config.wo[0]
            + self.point_object.normalized_redundancy * self.config.wo[1]
        )
        if self.point_object.all_element_batch_error_probabilities_normalized.is_cuda:
            self.point_object.redundancy_all_values_of_element_batch_normalized = (
                self.point_object.redundancy_all_values_of_element_batch_normalized.cuda()
            )

        self.point_object.all_element_batch_ponderate_objective = (
            self.point_object.all_element_batch_error_probabilities_normalized
            * self.config.wo[0]
            + self.point_object.redundancy_all_values_of_element_batch_normalized
            * self.config.wo[1]
        )
        self.point_object.wo = (
            self.point_object.probability_of_error_normalized * self.config.wo[0]
            + self.point_object.normalized_redundancy * self.config.wo[1]
        )

        epsilon = 1e-35
        minimum = torch.min(self.point_object.all_element_batch_ponderate_objective)
        maximum = torch.max(self.point_object.all_element_batch_ponderate_objective)
        self.point_object.all_element_batch_ponderate_objective_normalized = (
            self.point_object.weighted_objective - minimum
        ) / (maximum - minimum + epsilon)
