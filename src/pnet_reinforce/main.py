import logging

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

# ---- Local modules-------------------------
from config import get_config, print_config
from training.train import model_training
from evaluation.evaluation import model_evaluation
from utils.utils import load_weight, save_model
from network.actor.actor import Actor
from network.critic import Critic


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


    if config.train_mode:

        print("[INFO]: Traning model\n")
        if config.load_model:
            pointer_net, critic_net, opt_poiner, opt_critic = load_weight(
                config=config,
                pointer_net=pointer_net,
                critic_net=critic_net,
                opt_pointer=opt_critic,
                opt_critic=opt_critic,
                device=config.device,
            )

        else:
            print("[INFO] Learning from scratch with out no fine tunning")

        for i_epoch in tqdm(range(epochs)):
            pointer_net.train()
            critic_net.train()

            for i in range(stepsByEpochs):

                model_training(
                    pointer_net=pointer_net,
                    critic_net=critic_net,
                    opt_pointer=opt_pointer,
                    opt_critic=opt_critic,
                    MSEloss=MSEloss,
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
                model_evaluation(
                    pointer_net=pointer_net,
                    crictic_net=critic_net,
                    config=config,
                    plot=plot,
                    path=path,
                    is_training=True,
                )

        if config.save_model:
            save_model(config, pointer_net, critic_net, opt_pointer, opt_critic)

    else:
        print("[INFO]: Model evaluation")

        if config.load_model:
            pointer_net, critic_net, opt_poiner, opt_critic = load_weight(
                config, pointer_net, critic_net, opt_pointer, opt_critic, device
            )

        path = config.graphPath + "Evaluación"

        model_evaluation(
            pointer_net=pointer_net,
            critic_net=critic_net,
            config=config,
            plot=plot,
            path=path,
            printOpt=False,
        )


if __name__ == "__main__":
    main()
