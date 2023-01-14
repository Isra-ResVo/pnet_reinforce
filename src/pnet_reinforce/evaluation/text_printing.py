import torch
from evaluation.data_point import Point
from reward.reward_representation import DataSolution
def text_printing(point_object: Point, reward_grouped: DataSolution, index: int):
    text = {}
    text["pr_error"] = point_object.probability_of_error_no_log
    text["pr_ln"] = point_object.probability_of_error  # antes point['prLog']
    text["prErrorminimum"] = point_object.minimum_value_in_batch_element
    text["prErrormaximum"] = point_object.maximum_value_in_batch_element
    text["prMinAbs"] = torch.exp(point_object.minimum_value_in_batch_element)
    text["degPr"] = point_object.degradation_wrt_minimum
    text["won"] = point_object.all_element_batch_ponderate_objective_normalized

    text["redundancymax"] = point_object.redundancy_maximum
    text["redundancymin"] = point_object.redundancy_minimum
    text["redundancy"] = point_object.redundancy
    text["degR"] = point_object.redundancy_degradation_wrt_minimum

    text["n"] = point_object.n_inferred.item()
    text["k"] = point_object.k_inferred.item()
    text["redundancy"] = reward_grouped.redundancy[index]
    text["prn_ln"] = point_object.probability_of_error_normalized
    text["rn"] = reward_grouped.normalized_redundancy[index]
    text["prAbs"] = point_object.probability_of_error_no_log
    text["deg_pr_ln"] = text["prErrorminimum"] / text["pr_ln"]

    print2word = False
    if print2word:
        # Para escribir todo en latex

        fprint = "Salida ({n},{k}), n = {n}, k = {k} inferidos por el modelo \n\
                P_r max = {prErrormaximum:.3e}, P_r min = {prErrorminimum:.3e}, P_r = {pr_ln:.3e}\n\
                R max = {redundancymax}, R min = {redundancymin}, R = {redundancy:.2f}\n\
                WO min = {womin:.3f}, WO max = {womax:.3f}, WO = {wo:.3f}\n\
                P_{{rn}} = \\frac{{({pr_ln:.3e})-({prErrorminimum:.3e})}}{{({prErrormaximum:.3e})-({prErrorminimum:.3e})}}={prn_ln:.3f}\n\
                R_n = \\frac{{({redundancy:.1f})-({redundancymin:.1f})}}{{({redundancymax:.1f})-({redundancymin:.1f})}}={rn:.3f}\n\
                WO_n = \\frac{{({wo:.3f})-({womin:.3f})}}{{({womax:.3f})-({womin:.3f})}} = {won:.3f}\n\
                Deg(P_r) = \\frac{{({prAbs:.3e})}}{{({prMinAbs:.3e})}} = {degPr:.3e}, Deg(pr_{{ln}}) = \\frac{{({prErrorminimum:.3f})}}{{({pr_ln})}} = {deg_pr_ln:.3f} Deg(R) = \\frac{{({redundancy:.2f})}}{{(2.00)}} = {degR:.2f}\
            ".format(
            **text
        )
        print(fprint)
