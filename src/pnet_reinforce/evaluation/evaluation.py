import logging

import torch
import numpy as np

# from actor.actor import Reward
from reward.reward import Reward
from generator.evaluation import Evalution_batches
from extras import extraElements
from actor.actor import pr_vals_2_plot
from extras import plotting




def model_evaluation(
    pointer_net,
    critic_net,
    config,
    plot: float = False,
    path: str = None,
    printOpt: float = False,
    is_training: float = False,
):
    device = config.device

    print("valor de plot {}".format(plot))

    data_generator = Evalution_batches(config)

    if is_training:
        data_object = data_generator.item_batch_evalution(
            alternative_batchsize=1
        )

    else:
        
        print(
            "Replace element in memory: ",
            config.replace_element_in_memory,
        )
        data_object = data_generator.item_batch_evalution()


    data = {
        "len_elements": data_object.elements_length,
        "batch": data_object.batch,
        "indices": data_object.indices,
        "batchNormal": data_object.batch_normalized,
        "restricted_n": data_object.restricted_n,
        "restricted_k": data_object.restricted_k,
    }

    pointer_net.eval()
    critic_net.eval()

    logging.info("valores generados para el modelo:\n %s \n", str(data_object.batch))
    logging.info("indices del batch:\n\t %s", str(data_object.indices))

    if config.normal:
        batch = data_object.batch_normalized

        if config.extraElements == "elementsInBatch":
            oneElement = True if config.mode == "k" else False
            extraEle = extraElements(batch.shape, device, config.normal, oneElement)
            batch = torch.cat((batch, extraEle), dim=2)

    else:
        batch = data_object.batch

    # Inference
    selections, log_probs = pointer_net(batch, data)
    batch_steps = data_object.batch.shape[2]

    if config.mode == "k":
        reward = Reward(selections, device, batch_steps, config, data)
    else:
        reward = Reward(selections, device, batch_steps, config)

    rewardDict = reward.main(data)

    val_statistic = {
        "pr_ln": [],
        "prn_ln": [],
        "redundancy": [],
        "rn": [],
        "wo": [],
        "won": [],
    }

    print("reward dict this is used in the posteriori calculus", rewardDict["prError"])

    for index, _ in enumerate(batch):

        # Dict with values of first element of inference and print it in plot to see model performance
        point = {
            "value_error": rewardDict["prError"][index],
            "prError1": 0,
            "redundancy": rewardDict["redundancy"][index],
            "normRed": rewardDict["normRed"][index],
            # values to compare
            "n_position": reward.sel_n[index],
            "k_position": reward.k_sum[index],
            "onlyClouds": reward.only_clouds[index],
            "batchQntClouds": data["len_elements"][index],
        }

        # Elements to populate for plotting reasons

        toCompareInPlot = []
        tuples_data_and_names = []
        names = []
        text = {}

        if "prError" in config.whatToGraph:
            # cambio a peticion del Dr.
            # localNames = ['prError - Best Elements', 'prErorr - All combinations']
            local_name = ["Probabilidad de perdida"]  # revisar
            localNameToShow = ["prError"]  # revisar

            # inside of pr_val_2_plot various values of point dict are generated
            (
                tuple_pr_error_2graph,
                pr_normalized_plot,
                degPr,
                minimum,
                maximum,
            ) = pr_vals_2_plot(toCompareInPlot, data, point, config, device, index)
            text["pr_error"] = point["pr_error"]  # antes point['prAbs']
            text["pr_ln"] = point["pr_ln"]  # antes point['prLog']
            text["prErrorminimum"] = minimum
            text["prErrormaximum"] = maximum
            text["prMinAbs"] = torch.exp(minimum)
            text["degPr"] = degPr

            if plot:
                if config.mode == "n":
                    labeloptions = (True, "upper left")
                else:
                    labeloptions = (True, "lower left")

                pathLocal = path + " Probabilidad de perdida.png"

                if config.log:
                    points2graph = [(point["pr_ln"], "pr_ln")]
                    pr_error_2graph = tuple_pr_error_2graph[0]  # ln values

                else:
                    points2graph = [(point["pr_error"], "pr")]
                    pr_error_2graph = tuple_pr_error_2graph[1]

                data_and_name_2graph = [
                    (pr_error_2graph, local_name)
                ]  # before local names
                print("*" * 100)
                print(pr_error_2graph)
                plotting(
                    data_and_name_2graph,
                    points2graph,
                    point=point,
                    mode=config.mode,
                    path=pathLocal,
                    logScale=True,
                    labeloptions=labeloptions,
                )

                tuples_data_and_names.append((pr_normalized_plot, local_name))

        if "redundancy" in config.whatToGraph:
            localNames = ["Redundancia"]
            (
                redundancy,
                redundancy_original,
                degR,
                minimum,
                maximum,
            ) = reward.redundancyValsPlot(
                point=point, config=config, kwargs=data, index=index
            )
            text["redundancymax"] = maximum
            text["redundancymin"] = minimum
            text["redundancy"] = point["redundancy"]
            text["degR"] = degR

            if plot:
                if config.mode == "n":
                    labeloptions = (True, "upper right")
                else:
                    labeloptions = (True, "upper left")

                pathLocal = path + " Redundancia.png"
                points2graph = [(point["redundancy"], "redundancy")]
                data_and_name = [(redundancy_original, localNames)]
                plotting(
                    data_and_name,
                    points2graph,
                    point=point,
                    mode=config.mode,
                    path=pathLocal,
                    labeloptions=labeloptions,
                )

                tuples_data_and_names.append((redundancy, "Redundancia"))

        if pr_normalized_plot.is_cuda:
            redundancy = redundancy.cuda()

        woValues2Graph = pr_normalized_plot * config.wo[0] + redundancy * config.wo[1]
        # print('valores de woValues2Graph', elementToCompare)

        wo = point["prn_ln"] * config.wo[0] + point["normRed"] * config.wo[1]

        epsilon = 1e-35
        text["wo"] = wo
        text["womin"] = torch.min(woValues2Graph)
        text["womax"] = torch.max(woValues2Graph)
        text["won"] = (wo - text["womin"]) / (text["womax"] - text["womin"] + epsilon)

        point["wo"] = wo

        if plot:
            tuples_data_and_names.append((woValues2Graph, "WO"))
            pathComparation = path + " Comparacion de objetivos.png"
            annotatewo = False if 1 in config.wo else True
            # print('valor de annotate****************', annotatewo)

            if config.monoObjetive is not None:
                if config.monoObjetive == "prError":
                    points2graph = [(point["prn_ln"], "prn_nl")]
                elif config.monoObjetive == "redundancy":
                    points2graph = [(point["normRed"], "Rn")]
                else:
                    print(config.monoObjetive)
                    raise NotImplementedError(
                        "the value in monoObjetive is not valid only is implemented prError and redundancy"
                    )
            else:
                points2graph = [(point["wo"], "WO")]

            plotting(
                tuples_data_and_names,
                points2graph,
                point=point,
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
            print("\tn quantity: ", reward.sel_n[index])
            print("\tk quantity: ", reward.k_sum[index])
            print(
                "\t --->valor de comparacion (normRed+prError)/2: {}".format(
                    (point["normRed"] + point["value_error"]) / 2
                )
            )

            print("\n\nDifferent values in rewardDict")
            for key in rewardDict:
                print("\n\tkey element: {}".format(key))
                print("\t", rewardDict[key][index])

        print2word = False
        text["n"] = reward.sel_n[index].item()
        text["k"] = reward.k_sum[index].item()
        text["redundancy"] = rewardDict["redundancy"][index]
        text["prn_ln"] = point["prn_ln"]
        text["rn"] = rewardDict["normRed"][index]
        text["prAbs"] = point["pr_error"]
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
            val_statistic["pr_ln"].append(point["pr_ln"])
            val_statistic["redundancy"].append(rewardDict["normRed"][index])
            val_statistic["won"].append(text["won"])
            val_statistic["prn_ln"].append(text["prn_ln"])
            val_statistic["rn"].append(text["rn"])

    if config.statistic:
        print("valores adquiridos para la tupla")
        for i, (val_k, val_n) in enumerate(zip(reward.k_sum, reward.sel_n)):
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