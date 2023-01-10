import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# ---- Local modules-------------------------
from generator.evaluation import Evalution_batches
from actor import Actor
from actor import Reward
from actor import pr_vals_2_plot
from critic import Critic
from config import get_config, print_config
from extras import plotting
from extras import extraElements


def data_func(
    config,
    batchSize: int = None,
) -> dict:
    r"""

    Generate a dict with necessary data to train the model. This data is generated
    on the fly based in reports presented in another investigations...

    args
    ------
    shape_at_dist: str  [singleelement| batchelement]

    """

    data = {}

    new_generator = Evalution_batches(config)
    data_object = new_generator.item_batch_evalution(alternative_batchsize=batchSize)

    data["len_elements"] = data_object.elements_length
    data["batch"] = data_object.batch
    data["indices"] = data_object.indices
    data["batchNormal"] = data_object.batch_normalized
    data["restricted_n"] = data_object.restricted_n
    data["restricted_k"] = data_object.restricted_k

    return data


def train_epoch(
    pointer_net,
    critic_net,
    opt_pointer,
    opt_critic,
    MSEloss,
    schedulers,
    config,
):
    device = config.device
    # getting for training
    data = data_func(config=config)

    # control if the batch input to model is normalized (bool)
    if config.normal:
        batch = data["batchNormal"]

        if config.extraElements == "elementsInBatch":
            oneElement = True if config.mode == "k" else False
            extraEle = extraElements(batch.shape, device, config.normal, oneElement)
            batch = torch.cat((batch, extraEle), dim=2)

    else:
        batch = data["batch"]
    batch_steps = data["batch"].shape[2]

    # inferences
    selections, log_probs = pointer_net(batch, data)
    critic_pred = critic_net(
        batch, data
    )  # warning Revisar los elementos para evitar los elementos extra

    # Reward
    if config.mode == "k":
        reward = Reward(selections, device, batch_steps, config, data)
    else:
        reward = Reward(selections, device, batch_steps, config)

    rewardDict = reward.main(data)
    reward_baseline = (rewardDict[config.key_reward] - critic_pred.reshape(-1)).detach()

    loss_1 = torch.mean(reward_baseline * log_probs)
    loss_1.backward()
    opt_pointer.step()
    opt_pointer.zero_grad()

    loss_2 = MSEloss(critic_pred.reshape(-1), rewardDict[config.key_reward])
    loss_2.backward()
    opt_critic.step()
    opt_critic.zero_grad()


def system_evaluation(
    pointer_net,
    critic_net,
    config,
    plot=False,
    path: str = None,
    printOpt: float = False,
    evaluation: float = False,
):
    device = config.device

    print("valor de plot {}".format(plot))

    if evaluation:
        print(
            "Replace element in memory: ",
            config.replace_element_in_memory,
        )
        data = data_func(config=config)

    else:
        batchSize = 1
        data = data_func(config=config, batchSize=batchSize)

    pointer_net.eval()
    critic_net.eval()

    logging.info("valores generados para el modelo:\n %s \n", str(data["batch"]))
    logging.info("indices del batch:\n\t %s", str(data["indices"]))

    if config.normal:
        batch = data["batchNormal"]

        if config.extraElements == "elementsInBatch":
            oneElement = True if config.mode == "k" else False
            extraEle = extraElements(batch.shape, device, config.normal, oneElement)
            batch = torch.cat((batch, extraEle), dim=2)

    else:
        batch = data["batch"]

    # Inference
    selections, log_probs = pointer_net(batch, data)
    batch_steps = data["batch"].shape[2]

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


def load_weight(config, pointer_net, critic_net, opt_pointer, opt_critic, device):
    #  Cambiar esto por un ciclo for    warning
    print("\n [INFO]: Cargando pesos al modelo")

    weight_dict = torch.load(config.load_from, map_location=torch.device(device))

    pointer_net.load_state_dict(weight_dict["model_pointer"])
    critic_net.load_state_dict(weight_dict["model_critic"])

    if config.load_optim:
        print("[INFO] Cargando los parametros del optimizador")
        opt_pointer.load_state_dict(weight_dict["opt_pointer"])
        opt_critic.load_state_dict(weight_dict["opt_critic"])
    else:
        print("[INFO] No se cargando los parametros del optimizador")

    print(" [INFO]: Listo\n")

    return pointer_net, critic_net, opt_pointer, opt_critic


def save_model(config, pointer_net, critic_net, opt_pointer, opt_critic):
    torch.save(
        {
            "model_pointer": pointer_net.state_dict(),
            "model_critic": critic_net.state_dict(),
            "opt_pointer": opt_pointer.state_dict(),
            "opt_critic": opt_critic.state_dict(),
        },
        config.save_to,
    )

    print("[INFO] Success in save model")


def main():

    # Get default valus for the system
    print_config()
    config, _ = get_config()
    plot = config.plot

    # for debuggin
    if not config.train_mode and config.debugcomments:
        logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.device = device

    # Nerwork initialization
    pointer_net = Actor(config, device)
    critic_net = Critic(config, device)  # solo en su entrenamiento

    # Assingning models to Device GPU o CPU (Default)
    pointer_net.to(device)
    critic_net.to(device)

    # Optimizers instatiliztion
    opt_pointer = Adam(
        pointer_net.parameters(),
        lr=1e-03,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=config.weight_decay,
        amsgrad=False,
    )
    opt_critic = Adam(
        critic_net.parameters(),
        lr=1e-03,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=config.weight_decay,
        amsgrad=False,
    )
    MSEloss = nn.MSELoss()

    # Learning decay (model optimization) it doesn't work with verbose (Wrappers)
    schedulers = {
        "pointer": torch.optim.lr_scheduler.StepLR(
            opt_pointer, step_size=3, gamma=0.1, last_epoch=-1
        ),
        "critic": torch.optim.lr_scheduler.StepLR(
            opt_critic, step_size=3, gamma=0.1, last_epoch=-1
        ),
    }

    # Training params integrate this to config file to simplify
    epochs = config.num_epoch
    stepsByEpochs = 500

    # Trainin Mode
    if config.train_mode:

        print("[INFO]: Traning model\n")
        if config.load_model:
            # Load model pre-trained
            pointer_net, critic_net, opt_poiner, opt_critic = load_weight(
                config, pointer_net, critic_net, opt_pointer, opt_critic, device
            )

        else:
            print("[INFO] Learning from scratch with out no fine tunning")

        for i_epoch in tqdm(range(epochs)):
            pointer_net.train()
            critic_net.train()

            for i in range(stepsByEpochs):

                train_epoch(
                    pointer_net=pointer_net,
                    cretic_net=critic_net,
                    opt_pointer=opt_pointer,
                    opt_critic=opt_critic,
                    # generator,
                    MSEloss=MSEloss,
                    # device,
                    schedulers=schedulers,
                    config=config,
                )

                if (i % 200 == 0) and (i > 0):
                    print(" Iteration {} of {}".format(i, stepsByEpochs))

            # Learning rate ajust
            schedulers["pointer"].step()
            schedulers["critic"].step()
            print(
                "\n\n Learning rate in schedulers: \n     Pointer net: {}\n     \ Critic: {}".format(
                    schedulers["pointer"].get_last_lr(),
                    schedulers["critic"].get_last_lr(),
                )
            )

            # Control point
            if config.save_model and i_epoch % 1 == 0:
                print("control point (saving model's weight) {}".format(i_epoch + 1))
                save_model(config, pointer_net, critic_net, opt_pointer, opt_critic)

            # System evaluation a brief lookup to performance (function)
            evaluation = True  # agregare to config
            if evaluation and i_epoch % 1 == 0:
                path = "{} Epoch {}".format(config.graphPath, i_epoch)
                system_evaluation(
                    pointer_net=pointer_net,
                    crictic_net=critic_net,
                    config=config,
                    plot=plot,
                    path=path,
                )

        if config.save_model:
            save_model(config, pointer_net, critic_net, opt_pointer, opt_critic)

    # Inference mode for evaluation
    else:
        print("[INFO]: Model evaluation")

        if config.load_model:
            # warning Maybe change this with a dict with all the objects
            pointer_net, critic_net, opt_poiner, opt_critic = load_weight(
                config, pointer_net, critic_net, opt_pointer, opt_critic, device
            )

        path = config.graphPath + "Evaluación"
        diskelement2compare = config.shape_at_disk  # Bool
        # Model evaluation/inference fuction
        shape_at_disk = config.shape_at_disk  # str

        system_evaluation(
            pointer_net=pointer_net,
            critic_net=critic_net,
            config=config,
            plot=plot,
            path=path,
            printOpt=False,
            evaluation=True,
        )


if __name__ == "__main__":
    main()
