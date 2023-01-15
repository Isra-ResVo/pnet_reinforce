import torch

from generator.evaluation import Evalution_batches
from extras import extraElements

# from actor.actor import Reward
from reward.reward import Reward, RewardConfig
from network.critic import Critic


def model_training(
    pointer_net,
    critic_net: Critic,
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

    # control if the batch input to model is normalized (bool)
    if config.normal:
        batch = data_object.batch_normalized

        if config.extraElements == "elementsInBatch":
            oneElement = True if config.mode == "k" else False
            extraEle = extraElements(batch.shape, device, config.normal, oneElement)
            batch = torch.cat((batch, extraEle), dim=2)

    else:
        batch = data_object.batch
    batch_steps = data_object.batch.shape[2]

    # inferences
    selections, log_probs = pointer_net(x=batch, data_object=data_object)
    critic_pred = critic_net(x=batch, data_object=data_object)
    # warning Revisar los elementos para evitar los elementos extra

    # Reward
    reward_config = RewardConfig(
        selections=selections,
        device=device,
        qnt_steps=batch_steps,
        config=config,
        value_k=data_object.restricted_k,
    )
    reward = Reward(reward_config=reward_config)
    reward_grouped = reward.main(data_object=data_object)

    condition = config.objective_to_optimize
    if condition == "probability_of_error":
        objective = reward_grouped.probability_of_error
    elif condition == "normalized_pr_error":
        objective = reward_grouped.normalized_pr_error
    elif condition == "redundancy":
        objective = reward_grouped.redundancy
    elif condition == "normalized_redundancy":
        objective = reward_grouped.normalized_redundancy
    else:
        objective = reward_grouped.ponderate_objetive

    reward_baseline = (objective - critic_pred.reshape(-1)).detach()

    loss_1 = torch.mean(reward_baseline * log_probs)
    loss_1.backward()
    opt_pointer.step()
    opt_pointer.zero_grad()

    loss_2 = MSEloss(critic_pred.reshape(-1), objective)
    loss_2.backward()
    opt_critic.step()
    opt_critic.zero_grad()
