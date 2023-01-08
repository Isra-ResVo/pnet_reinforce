import numpy as np
import torch
from config import get_config
import pickle
import sys
import logging


def valuesF(self=None):

    """
    1 -- GoogleDrive
    2 -- OneDrive
    3 -- DropBox
    4 -- Box
    5 -- Egnyte
    6 -- Sharefile
    7 -- Salesforce
    8 -- Alibaba cloud
    9 -- Amazon Cloud Drive
    10 - Apple iCloud
    11 - Azure Storage
    """

    prob_fail = {
        0: (0.00072679, 0.00145361),
        1: (0.00066020, 0.00132043),
        2: (0.00097032, 0.00194065),
        3: (0.00179699, 0.00359402),
        4: (0.00073249, 0.00146503),
        5: (0.00014269, 0.00028542),
        6: (0.00061739, 0.00123480),
        7: (0.00079897, 0.00138793),
        8: (0.00018227, 0.00098241),
        9: (0.00015664, 0.00097632),
        10: (0.00039977, 0.00087241),
    }
    vel_download = {
        0: (2.15, 3.26),
        1: (1.21, 2.41),
        2: (3.07, 3.32),
        3: (2.01, 3.20),
        4: (2.17, 2.36),
        5: (0.72, 0.76),
        6: (0.68, 0.72),
        7: (2.54, 3.18),
        8: (2.49, 3.09),
        9: (2.01, 2.98),
        10: (2.30, 3.12),
    }
    vel_upload = {
        0: (1.79, 3.24),
        1: (0.91, 1.70),
        2: (2.59, 3.05),
        3: (1.91, 3.27),
        4: (1.24, 1.93),
        5: (0.11, 0.65),
        6: (0.52, 0.73),
        7: (2.32, 3.14),
        8: (0.70, 1.86),
        9: (2.05, 3.45),
        10: (1.31, 3.17),
    }

    return prob_fail, vel_download, vel_upload


def optimisticPessimistic(self=None, indices=None) -> np.ndarray:

    r"""
    Returs a batch with two elements:
        first element: contain the best probabilities of CSP
        second element: conatain the worst probabilities of CSP

    final shape: (2,1,number of CSP), to keep consistency with the rest of code
    """
    # Get the values the probabilities from a dict

    fail_prob = valuesF()[0]

    if indices is not None:
        print(indices)
        arg_iter = indices.cpu().numpy()
    else:
        arg_iter = fail_prob

    array = np.array([[] for _ in range(2)])

    for key in arg_iter:
        mini, maxi = fail_prob[key]
        array = np.concatenate((array, np.array([[mini], [maxi]])), axis=1)
    array = np.expand_dims(array, axis=1)
    array = np.sort(array, axis=-1)

    return array


class batchGenerator(object):
    r"""
    Generador de lotes para el entrenamiento
    las dimenciones serán la siguiente  (batch, channel (dims),  lengt) =
    (batch, data_dim, clouds)
    esto principalmente devido a que de esta manera lo pide pytorch

    funcion  sel_n
    Cuando de quiere fijar el a número especifico de nubes a las cuales seleccionar

    """
    # Funtions declareted outer of Class
    get_values = valuesF

    # Funtions declarated inside
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.parameters = config.parameters

    def random_number_in_range_of_len_elements(
        self, batch_size=None, len_elements=None
    ):
        # This function is the same for k_mask because k is almost n
        #   Reason: k <= n and for this n_clouds function is for k too.
        if self.config.variable_length:
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

    def create_new(self, path, batch_size=1, variable=False):
        r"""
        Take the first element from the batch and indexs
        """
        toSave = self.generate_elements_list(batch_size=batch_size, variable=variable)
        with open(file=path, mode="wb") as f:
            pickle.dump(toSave, f)
        print("New item created and saved in disk, path: {}".format(path))
        return toSave

    def itemEvaluation(self, create=False):
        variable_length = self.config.variable_length
        if variable_length:
            path = "../save/batch_proof_variable"
        else:
            path = "../save/batch_proof"
        batch_size = 1
        tuplebatchindices = self.batchstored(create, path, batch_size, variable_length)
        return tuplebatchindices

    def batchEvaluation(self, create=False):
        variable_length = self.config.variable_length
        if variable_length:
            path = "../save/batch_20elements_variable"
        else:
            path = "../save/batch_20elements"
        batch_size = 20
        tuplebatchindices = self.batchstored(create, path, batch_size, variable_length)
        return tuplebatchindices

    def batchstored(self, create: bool, path: str, batch_size=1, variable=False):

        r"""This function create a batch with an element and, is disk persistent.
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

    def generate_elements_list(self, batch_size=None, max_length=None, variable=False):
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

    def generate_batch(self, batch_size=None, max_length=None):

        """
        1 -- GoogleDrive
        2 -- OneDrive
        3 -- DropBox
        4 -- Box
        5 -- Egnyte
        6 -- Sharefile
        7 -- Salesforce
        8 -- Alibaba cloud
        9 -- Amazon Cloud Drive
        10 - Apple iCloud
        11 - Azure Storage
        """

        # get values in prob_fail, vel_download, vel_upload in that respective order
        parameters = self.get_values()
        parameters = parameters[
            : self.parameters
        ]  # parameters: pr_error, vel_download, vel_upload

        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.max_length

        # batch_size = 1
        batch = []
        batch_indices = []
        for idx in range(batch_size):
            indices = np.random.choice(a=11, size=(self.max_length,), replace=False)
            batch_indices.append(indices)
            elements = []
            for i in indices:
                element = []
                for parameter in parameters:
                    minimo, maximo = parameter[i]
                    element.append(
                        np.random.uniform(low=minimo, high=maximo, size=(1,))
                    )
                elements.append(np.concatenate(element, axis=None))  # -> [11,3]
            batch.append(np.stack(elements, axis=-1))  # -> [3, max_length]
        batch = np.stack(batch, axis=0)
        batch_indices = np.stack(batch_indices, axis=0)

        # eliminated self_config

        return batch, batch_indices

    def general_norm(self, parameters=1):
        fail_prob, vel_download, vel_upload = self.get_values()
        batch = np.random.uniform(
            0, 1, size=(self.batch_size, parameters, self.max_length + 1)
        )

        return batch

    def getnames(self, indices):
        names = {
            0: "GoogleDrive",
            1: "OneDrive",
            2: "DropBox",
            3: "Box",
            4: "Egnyte",
            5: "ShareFile",
            6: "SalesForce",
            7: "Alibaba cloud",
            8: " Amazon Cloud Drive",
            9: "Apple iCloud",
            10: "Azure Storage",
        }

        print("Names of provider")
        for i in indices:
            print("\tkey {}, CSP {}".format(i, names[i]))


def main():
    #     indicesbatchpermute = [2, 3, 8, 1, 0, 7, 9, 5, 4, 6]
    #     indices = sorted(indicesbatchpermute)
    #     config, _ = get_config()

    #     gen = batchGenerator(config)
    #     gen.getnames(indices)

    #     prob, _, _ = gen.get_values()

    #     print('\n\nvalores minimos')

    #     for i in indices:
    #         print('{}'.format(prob[i][0]))

    #     print('\n\nValores maximos')

    #     for i in indices:
    #         print('{}'.format(prob[i][1]))

    #     exitmodel = [9,  8,  7,  6,  4,  3,  5,  0,  2]

    #     rearange = []
    #     for i in exitmodel:
    #         rearange.append(indicesbatchpermute[i]+1)

    #     print('La salida del modelo: ', rearange)

    config, _ = get_config()
    gen = batchGenerator(config)
    val, index = gen.generate_elements_list(variable=config.variable_length)
    print("\ncontenido de val\n", val)
    print("\ncontenido de index\n", index)

    for (i, j) in zip(val, index):
        print("\n")
        print(i.shape)
        print(i, j.shape[0])

    len_elements = [i.shape[0] for i in index]

    output = gen.n_clouds(len_elements=len_elements)
    print(output)


if __name__ == "__main__":

    main()
