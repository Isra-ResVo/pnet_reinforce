import torch
import numpy as np
from utils.plotter_data_helper import pr_vals_2_plot
from extras import plotting
from reward.helpers import HelperPlottingPoints
from reward.reward_representation import DataSolution
from generator.data_interface.data import DataRepresentation

from evaluation.data_point import Point, PointReferences


def extend_information(
    data_object: DataRepresentation,
    reward_grouped: DataSolution,
    config,
    path: str,
    plot: bool,
    printOpt: float,
):
    val_statistic = {
        "pr_ln": [],
        "prn_ln": [],
        "redundancy": [],
        "rn": [],
        "wo": [],
        "won": [],
    }
    device = config.device
    batch = data_object.batch

    for index, _ in enumerate(batch):

        point_object = Point(
            reward_grouped=reward_grouped, data_object=data_object, index=index
        )

        toCompareInPlot = []
        tuples_data_and_names = []
        names = []
        text = {}

        if "prError" in config.whatToGraph:
            # cambio a peticion del Dr.
            # localNames = ['prError - Best Elements', 'prErorr - All combinations']
            local_name = ["Probabilidad de perdida"]  # revisar
            localNameToShow = ["prError"]  # revisar

            # add more data to the point.
            PointReferences(point=point_object).pr_vals_2_plot(
                data_object=data_object, config=config, idxEle=index
            )

            tuple_pr_error_2graph = (
                point_object.all_element_batch_error_probabilities,
                point_object.all_element_batch_error_probabilities_no_log,
            )

            text["pr_error"] = point_object.probability_of_error_no_log
            text["pr_ln"] = point_object.probability_of_error  # antes point['prLog']
            text["prErrorminimum"] = point_object.minimum_value_in_batch_element
            text["prErrormaximum"] = point_object.maximum_value_in_batch_element
            text["prMinAbs"] = torch.exp(point_object.minimum_value_in_batch_element)
            text["degPr"] = point_object.degradation_wrt_minimum

            if plot:
                if config.mode == "n":
                    labeloptions = (True, "upper left")
                else:
                    labeloptions = (True, "lower left")

                pathLocal = path + " Probabilidad de perdida.png"

                if config.log:
                    points2graph = [(point_object.probability_of_error, "pr_ln")]
                    pr_error_2graph = tuple_pr_error_2graph[0]  # ln values

                else:
                    points2graph = [(point_object.probability_of_error_no_log, "pr")]
                    pr_error_2graph = tuple_pr_error_2graph[1]

                data_and_name_2graph = [
                    (pr_error_2graph, local_name)
                ]  # before local names
                print("*" * 100)
                print(pr_error_2graph)
                plotting(
                    data_and_names=data_and_name_2graph,
                    pointsToGraph=points2graph,
                    point=point_object,
                    mode=config.mode,
                    path=pathLocal,
                    logScale=True,
                    labeloptions=labeloptions,
                )

                tuples_data_and_names.append(
                    (
                        point_object.all_element_batch_error_probabilities_normalized,
                        local_name,
                    )
                )

        if "redundancy" in config.whatToGraph:

            localNames = ["Redundancia"]
            helper_data_redundancy = HelperPlottingPoints(
                reward_config=reward_grouped.reward_config
            )
            helper_data_redundancy.redundancyValsPlot(
                point=point_object, config=config, data_object=data_object, index=index
            )

            redundancy = point_object.redundancy_all_values_of_element_batch_normalized

            text["redundancymax"] = point_object.redundancy_maximum
            text["redundancymin"] = point_object.redundancy_minimum
            text["redundancy"] = point_object.redundancy
            text["degR"] = point_object.redundancy_degradation_wrt_minimum

            if plot:
                if config.mode == "n":
                    labeloptions = (True, "upper right")
                else:
                    labeloptions = (True, "upper left")

                pathLocal = path + " Redundancia.png"
                points2graph = [(point_object.redundancy, "redundancy")]
                data_and_name = [
                    (point_object.redundancy_all_values_of_element_batch, localNames)
                ]

                plotting(
                    data_and_name,
                    points2graph,
                    point=point_object,
                    mode=config.mode,
                    path=pathLocal,
                    labeloptions=labeloptions,
                )

                tuples_data_and_names.append(
                    (
                        point_object.redundancy_all_values_of_element_batch_normalized,
                        "Redundancia",
                    )
                )

        if point_object.all_element_batch_error_probabilities_normalized.is_cuda:
            point_object.redundancy_all_values_of_element_batch_normalized = (
                point_object.redundancy_all_values_of_element_batch_normalized.cuda()
            )

        woValues2Graph = (
            point_object.all_element_batch_error_probabilities_normalized * config.wo[0]
            + point_object.redundancy_all_values_of_element_batch_normalized * config.wo[1]
        )
        # print('valores de woValues2Graph', elementToCompare)

        wo = (
            point_object.probability_of_error_normalized * config.wo[0]
            + point_object.normalized_redundancy * config.wo[1]
        )

        epsilon = 1e-35
        text["wo"] = wo
        text["womin"] = torch.min(woValues2Graph)
        text["womax"] = torch.max(woValues2Graph)
        text["won"] = (wo - text["womin"]) / (text["womax"] - text["womin"] + epsilon)

        point_object.weighted_objective = wo

        if plot:
            tuples_data_and_names.append((woValues2Graph, "WO"))
            pathComparation = path + " Comparacion de objetivos.png"
            annotatewo = False if 1 in config.wo else True
            # print('valor de annotate****************', annotatewo)

            if config.monoObjetive is not None:
                if config.monoObjetive == "prError":
                    points2graph = [
                        (point_object.probability_of_error_normalized, "prn_nl")
                    ]
                elif config.monoObjetive == "redundancy":
                    points2graph = [(point_object.normalized_redundancy, "Rn")]
                else:
                    print(config.monoObjetive)
                    raise NotImplementedError(
                        "the value in monoObjetive is not valid only is implemented prError and redundancy"
                    )
            else:
                points2graph = [(point_object.weighted_objective, "WO")]

            plotting(
                tuples_data_and_names,
                points2graph,
                point=point_object,
                mode=config.mode,
                path=pathComparation,
                annotatewo=annotatewo,
            )

        if printOpt:
            print(
                "\n\n  Resultados de la experimetacion, número de experimento {}".format(
                    index
                )
            )

            print("\n\nSelecton of tuple (k,n):")
            print("\tn quantity: ", reward_grouped.n_inferred[index])
            print("\tk quantity: ", reward_grouped.k_inferred[index])
            print(
                "\t --->valor de comparacion (normRed+prError)/2: {}".format(
                    (
                        point_object.normalized_redundancy
                        + point_object.probability_of_error_normalized
                    )
                    / 2
                )
            )

        print2word = False
        text["n"] = point_object.n_inferred.item()
        text["k"] = point_object.k_inferred.item()
        text["redundancy"] = reward_grouped.redundancy[index]
        text["prn_ln"] = point_object.probability_of_error_normalized
        text["rn"] = reward_grouped.normalized_redundancy[index]
        text["prAbs"] = point_object.probability_of_error_no_log
        text["deg_pr_ln"] = text["prErrorminimum"] / text["pr_ln"]
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

        if config.statistic:
            val_statistic["wo"].append(text["wo"])
            val_statistic["pr_ln"].append(point_object.probability_of_error)
            val_statistic["redundancy"].append(
                reward_grouped.normalized_redundancy[index]
            )
            val_statistic["won"].append(text["won"])
            val_statistic["prn_ln"].append(point_object.probability_of_error_normalized)
            val_statistic["rn"].append(text["rn"])

    if config.statistic:
        print("valores adquiridos para la tupla")
        for i, (val_k, val_n) in enumerate(
            zip(reward_grouped.k_inferred, reward_grouped.n_inferred)
        ):
            print(
                "Experimento no.{}, valor de la tupla({},{})".format(
                    i + 1, val_k, val_n
                )
            )

        for key in val_statistic:
            val_statistic[key] = torch.stack(val_statistic[key]).numpy()

            print("\nDato {}".format(key))
            if val_statistic[key].shape[0] <= 1:
                print(
                    "No se puede realizar operaciones estadisticas con un solo elemento"
                )
            else:
                print("Media, Varianza, Desviación estarndar")
                print(
                    "{:.4f}, {:.6f}, {:.4f}".format(
                        np.mean(val_statistic[key]),
                        np.var(val_statistic[key], ddof=1),
                        np.std(val_statistic[key], ddof=1),
                    )
                )
                print("Datos que se utilizaron para el calculo:\n", val_statistic[key])
