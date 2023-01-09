from abc import ABC
from pnet_reinforce.generator.generator.initial_data import (
    FAILURE_PROBABILITY,
    UPLOAD_SPEED,
    DOWLOAD_SPEED,
)


class BaseBatchGenerator(ABC):
    r"""

    Basic block of contruction and configuration
    for api method related to data based on a pre
    configuration.

    The minimum of information that the generator
    requires  is the information required by this
    class and this proper functioning.

    The config parameter must be the same be the same
    between calls.

    """

    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.parameters = config.parameters
        self.variable_length = config.variable_length
        self.item_in_memory = config.item_in_memory
        self.device = config.device
        self.config = config

        # data used to generate the distributions
        self.building_data = [FAILURE_PROBABILITY, DOWLOAD_SPEED, UPLOAD_SPEED]
