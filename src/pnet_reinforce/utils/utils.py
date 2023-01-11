import torch

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