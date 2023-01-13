import torch

from reward.reward_representation import DataSolution
from generator.data_interface.data import DataRepresentation
from reward.error_func import pr_error_bound

selected_error_formula = pr_error_bound


class Point(object):
    r"""
    This object is for the representation of one point of information to
    graph and make comparatin with other points of data.

    Remember the probabilities are in log. Becuase the probabilities too small
    overflow the float in the calcualations.
    """

    def __init__(
        self,
        reward_grouped: DataSolution,
        data_object: DataRepresentation,
        index: int,

    ):
        # Config data
        self.config = reward_grouped.config

        # Essential data
        # remember the probability of error is log.
        self.probability_of_error = reward_grouped.probability_of_error[index]
        self.normalized_redundancy = reward_grouped.normalized_redundancy[index]
        self.redundancy = reward_grouped.redundancy[index]
        self.n_inferred = reward_grouped.n_inferred[index]
        self.k_inferred = reward_grouped.k_inferred[index]
        self.selected_clouds = reward_grouped.selected_clouds[index]
        self.elements_length = data_object.elements_length[index]
        
        # Probability of error referential  vals
        #   This are provided with PointRefereces tool.
        self.probability_of_error_no_log = None #"pr_ln"
        self.probability_of_error_normalized = None  # "prn_ln"
        self.probability_of_error_no_log_normalized = None  # "prn"
        self.maximum_value_in_batch_element = None
        self.minimum_value_in_batch_element = None
        self.degradation_wrt_minimum = None

        #   Multiple points 
        self.all_element_batch_error_probabilities = None
        self.all_element_batch_error_probabilities_normalized = None
        self.all_element_batch_error_probabilities_no_log = None

        # Redundancy values referencial values
        
        self.redundancy_all_values_of_element_batch_normalized = None
        self.redundancy_all_values_of_element_batch = None
        self.redundancy_degradation_wrt_minimum = None
        self.redundancy_minimum = None
        self.redundancy_maximum = None


        # weighted point
        self.weighted_objective = None



class PointReferences(object):
    def __init__(self, point: Point):
        self.point = point
        self.selected_error_formula = selected_error_formula

    def pr_vals_2_plot(self, data_object: DataRepresentation, config, idxEle=0):
        device = config.device
        # agregar config
        # All this only works for one element!
        local_error_formula = self.selected_error_formula
        mode = config.mode
        log = config.log

        batch = data_object.batch
        indixes = data_object.indices
        buffer = []
        bestWorst = [False, True]

        if mode == "k_n":
            # batch_optiPessi = torch.tensor(optimisticPessimistic(indices = indixes[idxEle]))
            # for i, _ in enumerate(batch_optiPessi):
            #       buffer.append(all_combinations(batch = batch_optiPessi, idx_batch = i, device = device))
            for boolean in bestWorst:
                buffer.append(
                    self.__all_combinations(
                        batch,
                        config,
                        idxEle,
                        data_object=data_object,
                        device=device,
                        largest=boolean,
                    )
                )

        elif mode == "n":
            n = self.point.n_inferred
            selections = self.point.selected_clouds
            # Replicating the first element for the batch

            siz = n - 1
            batch = batch[idxEle].repeat(siz, 1, 1)

            # First elements is to generate all the possible solutions with that selections of clouds
            # Second element is to generate all the posbible solutions with the best elements in batch

            # onlyCloudsBatch = [selections.repeat(siz,1),  torch.topk(batch[idxEle][0],k = n, largest = False)[1].repeat(siz,1)]
            onlyCloudsBatch = []
            for boolean in bestWorst:
                if config.variable_length:
                    len_element = data_object.elements_length[idxEle]
                    onlyCloudsBatch.append(
                        torch.topk(
                            batch[idxEle, 0, :len_element], k=n, largest=boolean
                        )[1].repeat(siz, 1)
                    )
                else:
                    onlyCloudsBatch.append(
                        torch.topk(batch[idxEle][0], k=n, largest=boolean)[1].repeat(
                            siz, 1
                        )
                    )

            k = torch.arange(2, n + 1, dtype=torch.int64, device=device)
            subsets = n - k + 1

        elif mode == "k":

            k = self.point.n_inferred
            selections = self.point.selected_clouds

            if config.variable_length:
                qnt_clouds = data_object.elements_length[idxEle]
                len_element = data_object.elements_length[idxEle]
            else:
                qnt_clouds = batch.shape[2]

            siz = qnt_clouds - k + 1
            batch = batch[idxEle].repeat(siz, 1, 1)
            elementsToTake = torch.arange(
                k, qnt_clouds + 1, dtype=torch.int64, device=device
            )

            onlyCloudsBatch = []
            for config_topk in bestWorst:
                onlyClouds = []
                for quantity in elementsToTake:
                    if config.variable_length:
                        onlyClouds.append(
                            torch.topk(
                                batch[idxEle, 0, :len_element],
                                k=quantity,
                                largest=config_topk,
                            )[1]
                        )
                    else:
                        onlyClouds.append(
                            torch.topk(
                                batch[idxEle][0], k=quantity, largest=config_topk
                            )[1]
                        )
                onlyCloudsBatch.append(onlyClouds)

            subsets = torch.arange(1, siz + 1, dtype=torch.int64, device=device)

        else:
            raise ValueError("invalid mode fix it")

        if mode == "k" or mode == "n":
            for onlyCloud in onlyCloudsBatch:
                buffer.append(
                    local_error_formula(
                        batch=batch, subsets=subsets, only_clouds=onlyCloud
                    )
                )

        buffer = torch.stack(buffer)
        maximum = torch.max(buffer)
        minimum = torch.min(buffer)

        # Only graph the option with best elements[0] because there are too many combinations, the other
        # This for recomendation of Dr.
        buffer = buffer[0]

        pr_error_2graph_ln = buffer
        pr_error_2graph = torch.exp(buffer)

        # Note!!!: This probability is in natural log.
        degradation_wrt_minimum_val = self.point.probability_of_error / minimum

        # point['prErrorOri'] = point['value_error']
        # point["pr_error"] = torch.exp(self.point.probability_of_error)
        self.point.probability_of_error_no_log = torch.exp(
            self.point.probability_of_error
        )

        epsilon = 1e-35
        buffer_normalized = (buffer - minimum) / (
            maximum - minimum - epsilon
        )  # it's in log(ln)
        self.point.probability_of_error_normalized = (
            self.point.probability_of_error - minimum
        ) / (maximum - minimum + epsilon)
        # point["prn"] = (torch.exp(point["pr_ln"]) - torch.exp(minimum)) / (
        #     torch.exp(maximum) - torch.exp(minimum) + epsilon
        # )
        self.point.probability_of_error_no_log_normalized = (
            self.point.probability_of_error_no_log - torch.exp(minimum)
        ) / (torch.exp(maximum) - torch.exp(minimum) + epsilon)


        # for element in buffer:
        #     toCompareInPlot.append(element)

        self.point.all_element_batch_error_probabilities = pr_error_2graph_ln
        self.point.all_element_batch_error_probabilities_no_log = pr_error_2graph
        self.point.all_element_batch_error_probabilities_normalized = buffer_normalized
        self.point.maximum_value_in_batch_element = maximum
        self.point.minimum_value_in_batch_element = minimum
        self.point.degradation_wrt_minimum = degradation_wrt_minimum_val
        
    

    def __all_combinations(
        self,
        batch,
        config,
        idx_batch=0,
        data_object: DataRepresentation = None,
        device="cpu",
        largest=False,
    ) -> torch.tensor:
        r"""
        All the combinations for each n and its respectives k values,
        with all the providers taken in the for one element in the batch,
        for instance:
        if we have n = 3 in the batch,  then the conbination for tuple(n,k) are
            (2,2),(2,3),(3,3)....
        where  2 <= k  <=n
        So the values calculated are all the probabilities of error of those
        combinations.

        This function only work with one element in the batch becuase the
        the work I'll be too much if we consider all the all the elements.
        Also this is only feasible with an n relative small because the combinations
        are On^2.

        idx_batch: is to choose the element to work with, default = 0

        Note: This only work with one element

        """
        local_error_formula = selected_error_formula
        # agregar config and kwargs
        if config.variable_length:
            len_element = data_object.elements_length[idx_batch]
            cloud_qnts = torch.arange(start=2, end=len_element + 1, device=device)

        else:
            dim_2 = batch.shape[2]
            cloud_qnts = torch.arange(start=2, end=dim_2 + 1, device=device)

        ks = [
            torch.arange(start=2, end=cloud_qnt + 1, device=device)
            for cloud_qnt in cloud_qnts
        ]

        # calculating the values for all the combinations of error of the (k,n) posibles
        all_values = []
        for cloud_element, ks_element in zip(cloud_qnts, ks):
            k_sum = ks_element
            sel_n = cloud_element.repeat(k_sum.shape[0])
            subsets = sel_n - k_sum + 1

            # Best k elements
            if config.variable_length:
                len_element = data_object.elements_length[idx_batch]
                _, index = torch.topk(
                    input=batch[idx_batch, 0, :len_element],
                    k=cloud_element,
                    largest=largest,
                )  # probable bug

            else:
                _, index = torch.topk(
                    input=batch[idx_batch][0], k=cloud_element, largest=largest
                )  # probable bug

            only_clouds = index.repeat(
                k_sum.shape[0], 1
            )  # índices de lo elementos a selecionar
            batch_cloned = (
                batch[idx_batch].clone().unsqueeze(dim=0).repeat(k_sum.shape[0], 1, 1)
            )

            # one element error probability
            pr_error = local_error_formula(
                batch=batch_cloned, subsets=subsets, only_clouds=only_clouds
            )

            all_values.append(pr_error)
        all_values = torch.cat(all_values, dim=0)

        return all_values
