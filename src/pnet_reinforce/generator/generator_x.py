import pickle
import sys
from abc import ABC, abstractmethod

# dev dependencies
import torch
import numpy as np

# local modules
from pnet_reinforce.generator.generator.initial_data import CLOUD_NAMES


class BatchGenerator(object):
    r"""
    Batch generator to train the model.
    The dimenction are  (batch_dim, channel (dims), lenght) =
    (batch_dim, data_dim, id_cloud)

    These are the required data dimentionality in Pytorch layers
    that we are using it.

    """
    # Funtions declarated inside
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.parameters = config.parameters
        self.variable_length = config.variable_length

    # only outside
    def random_number_in_range_of_len_elements(
        self, batch_size=None, len_elements=None
    ) -> torch.Tensor:
        r'''
        When we need to create elements to train in the special case
        where the value of n or k is fixed is necessary to make a
        special restriction to avoid generate invalid solution. These
        elements are generated here and then used in the network as
        a mask.

        When we are training for n is necessary to select a `n` cloud of
        providers quantity. And the network only can choose the value of
        k freely.

        This function is the same for k because k is almost n
        Reason: k <= n and for this n_clouds function is for k too.

        '''
        if self.variable_length:
            if len_elements is None:
                raise ValueError("must provide len_elements with varible length")
            elements_to_restric_behavior = []
            for len_element in len_elements:
                elements_to_restric_behavior.append(
                    torch.randint(2, len_element + 1, size=(1,))
                )
            elements_to_restric_behavior = torch.cat(
                elements_to_restric_behavior, dim=0
            )

        else:
            if batch_size is None:
                batch_size = self.batch_size
            elements_to_restric_behavior = torch.randint(
                2, self.max_length + 1, size=(batch_size,)
            )

        return elements_to_restric_behavior

    # intern
    def create_new(self, path, batch_size=1, variable=False):
        r"""
        Take the first element from the batch and indexs
        """
        toSave = self.generate_elements_list(batch_size=batch_size, variable=variable)
        with open(file=path, mode="wb") as f:
            pickle.dump(toSave, f)
        print("New item created and saved in disk, path: {}".format(path))
        return toSave

    # outside
    def itemEvaluation(self, create: bool = False):
        variable_length = self.variable_length
        if variable_length:
            path = "./saved_batchs/single_element_variable"
        else:
            path = "./saved_batchs/single_element"
        batch_size = 1
        tuplebatchindices = self.batchstored(create, path, batch_size, variable_length)
        return tuplebatchindices

    # outside
    def batchEvaluation(self, create=False):
        variable_length = self.variable_length
        if variable_length:
            path = "./save/20_elements_variable"
        else:
            path = "./save/20_elements"
        batch_size = 20
        tuplebatchindices = self.batchstored(create, path, batch_size, variable_length)
        return tuplebatchindices

    # intern
    def batchstored(self, create: bool, path: str, batch_size=1, variable=False):

        r"""
        This function create a batch with an element and is disk persistent.
        If an item doesn't exist previously, create a new item"""

        if create:
            print("New instance for evaluate created")
            tuplebatchindices = self.create_new(path, batch_size, variable)

        else:
            try:
                with open(file=path, mode="rb") as f:
                    tuplebatchindices = pickle.load(f)
                print("Item recovered from disk")

            except (IOError, EOFError):
                print("File not found or empty file")
                tuplebatchindices = self.create_new(path, batch_size, variable)

            except:
                print("Unexpected error", sys.exc_info()[0])
                raise RuntimeError("unexpedted Error")

        return tuplebatchindices

    # indise outside
    def generate_elements_list(
        self, batch_size=None, max_length=None, variable=False
    ) -> np.ndarray:
        def for_secuence(iterable, parameters_min_max):
            elements_secuences = []
            index_elements = []

            for ele_len in elements_length:
                index = np.random.choice(a=11, size=(ele_len,), replace=False)
                index_elements.append(index)
                element = []
                for parameter in parameters_min_max:
                    parameter_val_buffer = []
                    for i in index:
                        low, high = parameter[i]
                        parameter_val_buffer.append(
                            np.random.uniform(low=low, high=high, size=(1,))
                        )
                    element.append(
                        np.concatenate(parameter_val_buffer, axis=0)
                    )  # -> (len(parameters), ele_len)
                elements_secuences.append(np.stack(element, axis=0))

            return elements_secuences, index_elements

        def padding_secuences(secuence_list, max_length):
            len_list = len(secuence_list)
            num_parameters = secuence_list[0].shape[0]
            matrix_output = np.zeros(shape=(len_list, num_parameters, max_length))
            for i, element in enumerate(secuence_list):
                len_element = element.shape[1]
                matrix_output[i, :, :len_element] = element
            return matrix_output

        parameters_min_max = self.get_values()
        parameters_min_max = parameters_min_max[
            : self.parameters
        ]  # Parameters: pr_error, vel_download, vel_upload
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.max_length

        if variable:
            elements_length = np.random.randint(
                low=2, high=max_length + 1, size=(batch_size,)
            )
            elements_secuences, index_elements = for_secuence(
                elements_length, parameters_min_max
            )
            elements_secuences = padding_secuences(elements_secuences, max_length)

        else:
            elements_length = [max_length for _ in range(batch_size)]
            elements_secuences, index_elements = for_secuence(
                elements_length, parameters_min_max
            )
            elements_secuences = np.stack(elements_secuences, axis=0)

        return elements_secuences, index_elements

    # not used
    def getnames(self, indices):
        names = CLOUD_NAMES

        print("Names of provider")
        for i in indices:
            print("\tkey {}, CSP {}".format(i, names[i]))
