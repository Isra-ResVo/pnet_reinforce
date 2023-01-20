import torch
from generator.generator.base import BaseBatchGenerator


class RandomGenerator(BaseBatchGenerator):
    def __init__(self, config):
        super(RandomGenerator, self).__init__(config)

    def __random_number_in_range_of_len_elements(
        self, batch_size: int = None, len_elements: iter = None
    ) -> torch.Tensor:
        """
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

        args
        --------------
        len_elements: this is the special case where the elements in the
        batch changes in lenght and for this reason some elements have
        padding CSP.

        """
        if self.variable_length:
            if len_elements is None:
                raise ValueError("Must provide len_elements with varible length")
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

    def restriction_data(
        self,
        batchSize,
        len_elements,
        default_restriction=10,
    ):
        if not self.item_in_memory:
            restriction_data = self.__random_number_in_range_of_len_elements(
                batchSize, len_elements
            ).to(self.device)
        else:
            if self.variable_length:
                restriction_data = self.__random_number_in_range_of_len_elements(
                    batchSize, len_elements
                ).to(self.device)
            else:
                restriction_data = torch.tensor(
                    [default_restriction], dtype=torch.int64, device=self.device
                )
                if self.config.shape_at_disk == "batchelements":
                    restriction_data.repeat(20)
        return restriction_data
