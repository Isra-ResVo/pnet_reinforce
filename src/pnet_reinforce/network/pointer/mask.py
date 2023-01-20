import torch


def pointer_mask(
    step, data_object, main_mask, main_selection, steps_qnt, device
) -> torch.Tensor:
    """
    This mask and the others modify the state of the mask to avoid choose
    elements previously choosen and invalid solutions.

    The masked items are determined by the True values.

    In the conditions
    if step<2:
        The first two iterations only are for enforce to choose
        two items (valid solution).
    if step <3:
        k token is masked because k can't be k>n'.
    else:
        first blocks k token if k=n' to avoid k>n' (1)
        second block k token if stop token is selected (2).

    """

    cloud_number = (
        -2
    )  # Dont cosider the token only the items (k_position, stop_position)
    k_position = -2  # relative position of k token.
    stop_idx = -1  # relative positon of stop token.

    if step < 2:
        mask_clouds = torch.zeros_like(main_mask)
        mask_clouds[:, cloud_number:] = 1
        mask_condition = mask_clouds + main_mask

    elif step < 3:
        mask_clouds = torch.zeros_like(main_mask)
        mask_clouds[:, cloud_number] = 1  # bloquea la k
        mask_condition = mask_clouds + main_mask

    else:

        # (1)
        selections = torch.stack(main_selection, dim=1)  # [batch,steps_secuenfce]
        qnt_k = torch.sum((selections == steps_qnt - 2), dim=1)
        qnt_stop = torch.sum((selections == steps_qnt - 1), dim=1)
        qnt_clouds = step - (qnt_k + qnt_stop)

        condition_k_present = qnt_k > 0
        block_q = torch.zeros_like(main_mask).to(device)
        block_q[:, :k_position] = 1
        mask_condition = torch.where(
            condition_k_present.reshape(-1, 1), block_q, main_mask
        )

        # (2)
        condition_k = qnt_k + 2 < qnt_clouds
        # bool values
        stop_condition = main_mask[:, stop_idx:].type(
            torch.bool
        ) + ~condition_k.reshape(-1, 1)
        mask_all = torch.zeros_like(main_mask).to(device)
        mask_all[:, :stop_idx] = 1
        mask_condition = torch.where(stop_condition, mask_all, mask_condition).type(
            torch.bool
        )

    return mask_condition


def mask_n(step, kwargs, main_mask, main_selection, steps_qnt, device) -> torch.Tensor:
    """
    This condition is used when the n' is explicitly suministrated by
    the user.

    (1) Only let select n' items

    (2) Enforce that the value k is k<=n'
    """
    cloud_number = -2  # only item are valid tokens are out 
    stop_idx = -1
    if step <= 1:
        mask_kn = torch.zeros_like(main_mask).to(device)
        mask_kn[:, cloud_number:] = 1

        mask_condition = main_mask + mask_kn

    else:

        # gettting states of main_mask
        selections = torch.stack(main_selection, dim=1)  # [batch,steps_secuenfce]

        qnt_k = torch.sum((selections == steps_qnt - 2), dim=1)
        qnt_stop = torch.sum((selections == steps_qnt - 1), dim=1)
        qnt_clouds = step - (qnt_k + qnt_stop)

        # (1) Only selection of clouds
        cloud_cond = (qnt_clouds >= kwargs["restricted_n"]).reshape(-1, 1)
        mask_clouds = torch.zeros_like(main_mask, dtype=torch.int64).to(device)
        mask_clouds[:, :cloud_number] = 1

        mask_extra = torch.zeros_like(main_mask, dtype=torch.int64).to(device)
        mask_extra[:, cloud_number:] = 1  # ????? revisar este

        mask_clouds = torch.where(cloud_cond, mask_clouds, mask_extra)
        mask_withOutK = main_mask
        mask_withOutK[:, -2] = 0
        mask_condition = mask_clouds + mask_withOutK

        # (2) k selection with k <= n and q condition
        k_condition = (qnt_k + 2) >= kwargs["restricted_n"]

        q_condition = main_mask[:, stop_idx].type(torch.bool)
        condition_qk = (k_condition + q_condition).reshape(-1, 1)  # k or q (bool)

        mask_qk = torch.ones_like(main_mask).to(device)
        mask_qk[:, stop_idx] = 0  # only stop condition available

        mask_condition = torch.where(condition_qk, mask_qk, mask_condition)

    return mask_condition.type(torch.bool)


def mask_k(step, kwargs, main_mask, main_selection, steps_qnt, device) -> torch.Tensor:
    '''
    This mask used in the case that k is suministrated by the user
    so a minimum k item must to be choosen.
    
    '''

    stop_idx = -1
    # Validation k elements greater or equal to 2
    allElementsGreaterEqual2 = torch.any(kwargs["restricted_k"] >= 2)
    if not allElementsGreaterEqual2:
        raise ValueError("invalid k paramer")

    if step <= 1:  # Take minimum 2 elements in the batch
        mask_stopElement = torch.zeros_like(main_mask, device=device)
        mask_stopElement[:, stop_idx] = 1

        mask_condition = main_mask + mask_stopElement

    else:
        mask_stopElement = torch.zeros_like(main_mask, device=device)
        mask_stopElement[:, stop_idx] = 1
        maskPlusBlockStop = main_mask + mask_stopElement

        stopByItemsSelected = (
            step >= kwargs["restricted_k"]
        )  # This allows to grab items with minimum requirement of k items previosly selected.

        # mask that block  the stopElement  if  n < k
        mask_condition = torch.where(
            stopByItemsSelected.reshape(-1, 1), main_mask, maskPlusBlockStop
        )  # (condition, if true, else)

        conditionStopElement = main_mask[:, stop_idx].type(torch.bool).reshape(-1, 1)

        mask_clouds = torch.ones_like(main_mask, device=device)
        mask_clouds[:, -1] = 0

        mask_condition = torch.where(conditionStopElement, mask_clouds, mask_condition)

    return mask_condition.type(torch.bool)
