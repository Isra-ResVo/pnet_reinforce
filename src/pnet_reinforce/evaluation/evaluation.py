import logging

import torch
import numpy as np

# from actor.actor import Reward
from reward.reward import Reward, RewardConfig
from generator.evaluation import Evalution_batches
from extras import extraElements

from evaluation.extends import extend_information


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
        data_object = data_generator.item_batch_evalution(alternative_batchsize=1)

    else:

        print(
            "Replace element in memory: ",
            config.replace_element_in_memory,
        )
        data_object = data_generator.item_batch_evalution()

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
    selections, log_probs = pointer_net(batch, data_object)
    batch_steps = data_object.batch.shape[2]

    reward_config = RewardConfig(
        selections=selections,
        device=device,
        qnt_steps=batch_steps,
        config=config,
        value_k=data_object.restricted_k,
    )
    reward = Reward(reward_config=reward_config)
    reward_grouped = reward.main(data_object=data_object)

    extend_information(
        data_object=data_object,
        reward_grouped=reward_grouped,
        config=config,
        path=path,
        plot=plot,
        printOpt=printOpt,
    )
