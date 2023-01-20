import numpy as np
from generator.generator.base import BaseBatchGenerator


class ListElements(BaseBatchGenerator):
    def __init__(self, config):
        super(ListElements, self).__init__(config)

    def __for_secuence(self, elements_length, parameters_min_max):
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

    def __padding_secuences(self, secuence_list, max_length):
        len_list = len(secuence_list)
        num_parameters = secuence_list[0].shape[0]
        matrix_output = np.zeros(shape=(len_list, num_parameters, max_length))
        for i, element in enumerate(secuence_list):
            len_element = element.shape[1]
            matrix_output[i, :, :len_element] = element
        return matrix_output

    def generate_elements_list(
        self,
        batch_size: int = None,
        max_length: int = None,
    ):
        """
        This funtion is the main function to generate a batch data on the fly
        has two variants which are trigger with @variable var.

        1- The first option only generate a batch with n clouds(max_lenght)
        in all the elements in a ramdom way. Example: we want to generate a
        batch with dimentions (2,1,5) this will generate something like
        [
            [1,2,3,4,5]
            [1,2,3,4,5]
        ]

        2- The second option generates variable lenght elements With a minimum
        lenght of 2 and a miximum of max_lenght when a elements is < max_lenght
        then it is padding with 0 until reach the max_lenght. Example: we want
        to genearate batch with dimention (3,1,5) it will genreate something like
        {
            [1,2,3,0,0]
            [1,2,0,0,0]
            [1,2,3,4,5]
        }

        # Note that the numbers in the examples are only a representation and these
        are in reality a the probabilities or other kind of data.

        """
        # esay way to only requir the first data
        parameters_min_max = self.building_data[: self.parameters]

        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.max_length

        if self.variable_length:
            elements_length = np.random.randint(
                low=2, high=max_length + 1, size=(batch_size,)
            )
            elements_secuences, index_elements = self.__for_secuence(
                elements_length, parameters_min_max
            )
            elements_secuences = self.__padding_secuences(elements_secuences, max_length)

        else:
            elements_length = [max_length for _ in range(batch_size)]
            elements_secuences, index_elements = self.__for_secuence(
                elements_length, parameters_min_max
            )
            elements_secuences = np.stack(elements_secuences, axis=0)

        return elements_secuences, index_elements
