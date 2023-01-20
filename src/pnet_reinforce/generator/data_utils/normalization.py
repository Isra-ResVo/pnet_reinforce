import torch

def normalization(batch: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Normalize function -> (0,1)
    using the next formula by element and caracteristic
        element - minimum
        -----------------
        maximum - minimum

    this works with tensor arrange of the next way
        [batch index, caracteristic, elements]


    Probar: el normalizado entre un rango fijo de valores donde no se tome el maximo y el minomo de cada
    instancia para tenerlos como litmires naturales del problema

    """

    batchNorm = torch.zeros_like(batch)  # default device of input
    for i, element in enumerate(batch):
        for j, parameter in enumerate(element):
            mini = min(parameter)
            maxi = max(parameter)
            batchNorm[i][j] = (parameter - mini) / (maxi - mini)

    return batchNorm