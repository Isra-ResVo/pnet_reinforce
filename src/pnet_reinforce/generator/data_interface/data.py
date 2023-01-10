from abc import ABC
from generator.data_utils.normalization import normalization


class BaseDataStructure(ABC):
    def __init__(self, config):
        self.normalized_required: bool = config.normal
        self.normalization = normalization


class DataRepresentation(BaseDataStructure):
    r"""
    This object is to have all the data and variations
    that are necessary to some subproblems like k and n
    with fixed values.
    """

    def __init__(self, batch, indices, config):
        super(DataRepresentation, self).__init__(config=config)
        # base inforamtion
        self.batch = batch
        self.indices = indices
        self.elements_length = [len(element) for element in indices]
        self.batch_normalized = self.normalization(batch) if self.normalized_required else None
        
        

