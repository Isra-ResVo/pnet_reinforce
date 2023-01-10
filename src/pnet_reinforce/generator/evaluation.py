import sys
import pickle

import numpy as np
import torch

from generator.generator.base import BaseBatchGenerator
from generator.generator.list_elements import ListElements
from generator.data_utils.normalization import normalization
from generator.data_interface.data import DataRepresentation


class DataToDevice(BaseBatchGenerator):
    def __init__(self, config):
        super(DataToDevice, self).__init__(config)

    def assign_type_and_device(self, batch, indices):

        if self.variable_length:
            # All the elements could has different shapes
            indices = map(
                lambda x: torch.tensor(x, device=self.device, dtype=torch.int64),
                indices,
            )
        else:
            # All the elements has the same shapes
            indices = np.array(indices)
            indices = torch.tensor(indices, device=self.device, dtype=torch.int64)

        batch = torch.tensor(batch, device=self.device, dtype=torch.float32)
        return batch, indices


class Evalution_batches(BaseBatchGenerator):
    r"""
    This function constains method to retrieve data from memory to
    make evalutions and compare results. If the values doesn't exit
    in memory they are created with the help of generator class.

    Exist two main method to use externally the method to evaluate:
    1- item_evalution: single elements evaluation.
    2- batch_evalution: batch of 20 elements to evaluate.

    """

    def __init__(self, config):
        super(Evalution_batches, self).__init__(config=config)
        self.list_elements = ListElements(config=config)
        self.dataToDevice = DataToDevice(config=config)
        self.normalization = normalization

    def item_batch_evalution(
        self, alternative_batchsize: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        This function only manages the logic in how to provide the data
        based in the configuration. This configuration is based on the
        shape of elements (multi elements or single element), if use
        the item in memory or ignore it and use the predifined batch shape.

        """
        if (
            self.shape_at_disk == "singleelement"
            and self.item_in_memory
            and not self.train_mode
        ):
            batch, indices = self.__item_evaluation()
        elif (
            self.shape_at_disk == "batchelements"
            and self.item_in_memory
            and not self.train_mode
        ):
            batch, indices = self.__batch_evaluation()
        else:
            # this is used in traning and when evalution doesn't require to use
            # elements in memory.
            batch, indices = self.list_elements.generate_elements_list(
                batch_size=alternative_batchsize
            )

        batch, indices = self.dataToDevice.assign_type_and_device(batch, indices)

        data = DataRepresentation(
            batch=batch,
            indices=indices,
            config=self.config,
            alternative_batchsize=alternative_batchsize,
        )

        return data

    def __item_evaluation(self):
        variable_length = self.variable_length
        create = self.replace_element_in_memory
        if variable_length:
            path = "./saved_batchs/single_element_variable"
        else:
            path = "./saved_batchs/single_element"
        batch_size = 1
        tuplebatchindices = self.__batchstored(
            create, path, batch_size, variable_length
        )
        return tuplebatchindices

    def __batch_evaluation(self):
        variable_length = self.variable_length
        create = self.replace_element_in_memory
        if variable_length:
            path = "./saved_batchs/20_elements_variable"
        else:
            path = "./saved_batchs/20_elements"
        BATCH_SIZE = 20
        tuplebatchindices = self.__batchstored(
            create, path, BATCH_SIZE, variable_length
        )
        return tuplebatchindices

    def __create_new(self, path, batch_size=1):
        r"""

        Take the first element from the batch and indexs

        """
        toSave = self.list_elements.generate_elements_list(batch_size=batch_size)
        with open(file=path, mode="wb") as f:
            pickle.dump(toSave, f)
        print("New item created and saved in disk, path: {}".format(path))
        return toSave

    def __batchstored(
        self, create: bool, path: str, batch_size=1
    ) -> tuple[np.ndarray, np.ndarray]:

        r"""
        This method is a helper to manege the load or creation of
        new batches to evalution. If the `creation` is `False` then
        the helper will try to load the batch saved in memory and if this
        fails will create a new batch and saved in the correct path.

        """

        if create:
            print("New instance for evaluate created")
            tuplebatchindices = self.__create_new(path, batch_size)

        else:
            try:
                with open(file=path, mode="rb") as f:
                    tuplebatchindices = pickle.load(f)
                print("Item recovered from disk")

            except (IOError, EOFError):
                print("File not found or empty file")
                tuplebatchindices = self.__create_new(path, batch_size)

            except:
                print("Unexpected error", sys.exc_info()[0])
                raise RuntimeError("Unexpected error")

        return tuplebatchindices
