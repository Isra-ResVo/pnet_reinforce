from abc import ABC
from generator.data_utils.normalization import normalization
from generator.specialcases import RandomGenerator


class BaseDataStructure(ABC):
    def __init__(self, config):
        self.mode = config.mode
        self.item_in_memory = config.item_in_memory
        self.normalized_required: bool = config.normal
        self.normalization = normalization
        self.random_generator = RandomGenerator(config)


class DataRepresentation(BaseDataStructure):
    """
    This object is to have all the data and variations
    that are necessary to some subproblems like k and n
    with fixed values.
    """

    def __init__(self, batch, indices, config, alternative_batchsize=None):
        super(DataRepresentation, self).__init__(config=config)
        # base inforamtion
        self.batch = batch
        self.indices = indices
        self.elements_length = [len(element) for element in indices]
        self.batch_normalized = (
            self.normalization(batch) if self.normalized_required else None
        )
        self.restricted_n = None
        self.restricted_k = None

        if self.mode == "n" and not self.item_in_memory:
            self.restricted_n = self.random_generator.restriction_data(
                batchSize=alternative_batchsize,
                len_elements=self.elements_length,
                default_restriction=10,
            )
        elif self.mode == "k" and not self.item_in_memory:
            self.restricted_k = self.random_generator.restriction_data(
                batchSize=alternative_batchsize,
                len_elements=self.elements_length,
                default_restriction=2,
            )
