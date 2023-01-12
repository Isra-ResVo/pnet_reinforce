from reward.reward_representation import DataSolution
from generator.data_interface.data import DataRepresentation


class Point(object):
    r"""
    This object is for the representation of one point of information to
    graph and make comparatin with other points of data
    """

    def __init__(
        self, reward_grouped: DataSolution, data_object: DataRepresentation, index: int
    ):
        # Essential data
        self.probability_of_error = (reward_grouped.probability_of_error[index],)
        self.normalized_redundancy = (reward_grouped.normalized_redundancy[index],)
        self.redundancy = (reward_grouped.redundancy[index],)
        self.n_inferred = (reward_grouped.n_inferred[index],)
        self.k_inferred = (reward_grouped.k_inferred[index],)
        self.selected_clouds = (reward_grouped.selected_clouds[index],)
        self.elements_length = (data_object.elements_length[index],)
