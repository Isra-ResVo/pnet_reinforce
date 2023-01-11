import torch

from generator.evaluation import Evalution_batches
from extras import extraElements
# from actor.actor import Reward
from reward.reward import Reward, RewardConfig


def model_training(
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

    new_generator = Evalution_batches(config)
    data_object = new_generator.item_batch_evalution()
    data = {
        "len_elements": data_object.elements_length,
        "batch": data_object.batch,
        "indices": data_object.indices,
        "batchNormal": data_object.batch_normalized,
        "restricted_n": data_object.restricted_n,
        "restricted_k": data_object.restricted_k,
    }

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

    reward_config = RewardConfig(
        selections=selections,
        device=device,
        qnt_steps=batch_steps,
        config=config,
        value_k=data["restricted_k"],
    )

    # Reward
    if config.mode == "k":
        reward = Reward(reward_config=reward_config)
    else:
        reward = Reward(reward_config=reward_config)

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
