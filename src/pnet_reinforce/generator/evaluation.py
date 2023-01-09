import sys
import pickle
from pnet_reinforce.generator.generator.list_elements import ListElements



class Evalution_batches(object):
    r"""
    This function constains method to retrieve data from memory to
    make evalutions and compare results. If the values doesn't exit
    in memory they are created with the help of generator class.

    Exist two main method to use externally the method to evaluate:
    1- item_evalution: single elements evaluation.
    2- batch_evalution: batch of 20 elements to evaluate.

    """

    def __init__(self, config):
        self.list_elements = ListElements(config=config)


    def item_evaluation(self, variable_length: bool, create: bool = False):
        if variable_length:
            path = "./saved_batchs/single_element_variable"
        else:
            path = "./saved_batchs/single_element"
        batch_size = 1
        tuplebatchindices = self.batchstored(create, path, batch_size, variable_length)
        return tuplebatchindices

    def batch_evaluation(self, variable_length: bool, create: bool = False):
        if variable_length:
            path = "./saved_batchs/20_elements_variable"
        else:
            path = "./saved_batchs/20_elements"
        BATCH_SIZE = 20
        tuplebatchindices = self.batchstored(create, path, BATCH_SIZE, variable_length)
        return tuplebatchindices

    def create_new(self, path, batch_size=1, variable=False):
        r"""

        Take the first element from the batch and indexs
        
        """
        toSave = self.list_elements.generate_elements_list(batch_size=batch_size, variable=variable)
        with open(file=path, mode="wb") as f:
            pickle.dump(toSave, f)
        print("New item created and saved in disk, path: {}".format(path))
        return toSave

    # intern
    def batchstored(self, create: bool, path: str, batch_size=1, variable=False):

        r"""
        This method is a helper to manege the load or creation of 
        new batches to evalution. If the `creation` is `False` then 
        the helper will try to load the batch saved in memory and if this
        fails will create a new batch and saved in the correct path.

        """

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
                raise RuntimeError("Unexpected error")

        return tuplebatchindices
