import torch
from reward.reward import pr_error_bound
# Funtions declarated  to Reward class in outer scope
# Caution be aware with the self argument, bacaise is the first argument  from the system
selected_error_formula = pr_error_bound

def pr_vals_2_plot(toCompareInPlot, kwargs, point, config, device, idxEle=0):
    # agregar config
    # All this only works for one element!
    local_error_formula = selected_error_formula
    mode = config.mode
    log = config.log

    batch = kwargs["batch"]
    indixes = kwargs["indices"]
    buffer = []
    bestWorst = [False, True]

    if mode == "k_n":
        # batch_optiPessi = torch.tensor(optimisticPessimistic(indices = indixes[idxEle]))
        # for i, _ in enumerate(batch_optiPessi):
        #       buffer.append(all_combinations(batch = batch_optiPessi, idx_batch = i, device = device))
        for boolean in bestWorst:
            buffer.append(
                all_combinations(
                    batch, config, idxEle, kwargs, device=device, largest=boolean
                )
            )

    elif mode == "n":
        n = point["n_position"]
        selections = point["onlyClouds"]

        # Replicating the first element for the batch

        siz = n - 1
        batch = batch[idxEle].repeat(siz, 1, 1)

        # First elements is to generate all the possible solutions with that selections of clouds
        # Second element is to generate all the posbible solutions with the best elements in batch

        # onlyCloudsBatch = [selections.repeat(siz,1),  torch.topk(batch[idxEle][0],k = n, largest = False)[1].repeat(siz,1)]
        onlyCloudsBatch = []
        for boolean in bestWorst:
            if config.variable_length:
                len_element = kwargs["len_elements"][idxEle]
                onlyCloudsBatch.append(
                    torch.topk(batch[idxEle, 0, :len_element], k=n, largest=boolean)[
                        1
                    ].repeat(siz, 1)
                )
            else:
                onlyCloudsBatch.append(
                    torch.topk(batch[idxEle][0], k=n, largest=boolean)[1].repeat(siz, 1)
                )

        k = torch.arange(2, n + 1, dtype=torch.int64, device=device)
        subsets = n - k + 1

    elif mode == "k":

        k = point["k_position"]
        selections = point["onlyClouds"]

        if config.variable_length:
            qnt_clouds = kwargs["len_elements"][idxEle]
            len_element = kwargs["len_element"][idxEle]
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
                        torch.topk(batch[idxEle][0], k=quantity, largest=config_topk)[1]
                    )
            onlyCloudsBatch.append(onlyClouds)

        subsets = torch.arange(1, siz + 1, dtype=torch.int64, device=device)

    else:
        raise ValueError("imvalid mode fix")

    if mode == "k" or mode == "n":
        for onlyCloud in onlyCloudsBatch:
            buffer.append(
                local_error_formula(batch=batch, subsets=subsets, only_clouds=onlyCloud)
            )

    buffer = torch.stack(buffer)
    maximum = torch.max(buffer)
    minimum = torch.min(buffer)

    # Only graph the option with best elements[0] because there are too many combinations, the other
    # it's out for tutur recomendation.
    buffer = buffer[0]

    pr_error_2graph_ln = buffer
    pr_error_2graph = torch.exp(buffer)

    degradation = point["value_error"] / minimum
    # point['prErrorOri'] = point['value_error']
    point["pr_ln"] = point["value_error"]
    point["pr_error"] = torch.exp(point["value_error"])

    epsilon = 1e-35
    buffer_normalized = (buffer - minimum) / (
        maximum - minimum - epsilon
    )  # it's in log(ln)
    point["prn_ln"] = (point["pr_ln"] - minimum) / (maximum - minimum + epsilon)
    point["prn"] = (torch.exp(point["pr_ln"]) - torch.exp(minimum)) / (
        torch.exp(maximum) - torch.exp(minimum) + epsilon
    )

    # no log values

    # point['prError'] debe de ser point['prn']
    # point['prAbs'] debe de ser point['prError']
    # point['prlog'] debe de ser point['pr_ln']

    # for element in buffer:
    #     toCompareInPlot.append(element)

    return (
        (pr_error_2graph_ln, pr_error_2graph),
        buffer_normalized,
        degradation,
        minimum,
        maximum,
    )


def all_combinations(
    batch, config, idx_batch=0, kwargs=None, device="cpu", largest=False
) -> torch.Tensor:
    r"""
    All the combinations for k and n, with all the providers taken in the batch, for instance:
    if we have n = 3 in the batch,  then the conbination for tuple(n,k) are
        (2,2),(2,3),(3,3)....
    where  2 <= k  <=n

    idx_batch: is to choose the element to work with, default = 0

    Note: This only work with one element

    """
    local_error_formula = selected_error_formula
    # agregar config and kwargs
    if config.variable_length:
        len_element = kwargs["len_elements"][idx_batch]
        cloud_qnts = torch.arange(start=2, end=len_element + 1, device=device)

    else:
        dim_2 = batch.shape[2]
        cloud_qnts = torch.arange(start=2, end=dim_2 + 1, device=device)

    ks = [
        torch.arange(start=2, end=cloud_qnt + 1, device=device)
        for cloud_qnt in cloud_qnts
    ]

    # calculating the values for all the conbination of error tupole (k,n) posible
    all_values = []
    for cloud_element, ks_element in zip(cloud_qnts, ks):
        k_sum = ks_element
        sel_n = cloud_element.repeat(k_sum.shape[0])
        subsets = sel_n - k_sum + 1

        # Best k elements
        if config.variable_length:
            len_element = kwargs["len_elements"][idx_batch]
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
