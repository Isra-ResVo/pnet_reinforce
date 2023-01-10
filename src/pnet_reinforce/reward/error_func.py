import torch
import itertools

def error_function(
    self=None, batch=None, subsets: "list" = None, only_clouds: "list" = None
) -> "float":
    # Note this function uses itertools.combinations and for this reason could be too computationally expensive

    r"""
    Here the variables subsets and only clouds are dtype list
    subsets: contains list for selec j items (j = n-k+1 for every batch element)
        Example:
            [2,3,4]
    only_clouds: contain a list with index of elements, to be drag from batch values.
            [[0,3,4,5],
            [0,3,4,5],
            [0,3,4,5]]

    Note: very inefficient way to calculate this is better to take the values and make a bound
    because the complexity is too high for this porpouse
    """

    pr_error = []
    for idx_batch, (subset, clouds) in enumerate(zip(subsets, only_clouds)):

        clouds = torch.sort(clouds)[
            0
        ]  # This is necesary to get the same result (float point operations)
        combinations = itertools.combinations(clouds, subset)  # -> iter_object
        combinations = torch.stack(
            [torch.stack(comb) for comb in combinations]
        )  # -> Tensor

        buffer_pr_error = 0

        for combination in combinations:

            A_valuesT = batch[idx_batch, 0, combination]
            A_valores = torch.prod(A_valuesT)

            # Getting the complement values for combinations.
            mask = ~torch.sum(clouds == combination.reshape(-1, 1), dim=0).type(
                torch.bool
            )
            complement_A = clouds[mask]
            A_comp_values = torch.prod((1 - batch[idx_batch, 0, complement_A]))
            Pr_sub = A_valores * A_comp_values
            buffer_pr_error += Pr_sub

        pr_error.append(buffer_pr_error)
    pr_error = torch.stack(pr_error)
    pr_error = torch.log(pr_error)  # to avoid change a lot of code

    return pr_error


def pr_error_bound(
    self=None, batch=None, subsets: list = None, only_clouds: list = None
) -> torch.Tensor:

    r"""
    This function calculate the probability of failure by taking the combinations of
    elected clouds quantity by the model.

    This bound greater (>) than original formula, and is almost O(n) in complexity,
    converserly the original it is almost O(n^k).

    The main idea is only to use binomial coefficient to get all the possible combinations and
    multiply it by the worst cloud stats among the all them.

    """

    def factorial2(tensor):
        return (tensor + 1).to(torch.float32).lgamma().exp()

    def log_binomial_coefficient(n, k):
        first_exp = factorial2(n)
        second_exp = factorial2(k) * factorial2(n - k)
        return torch.log(first_exp / second_exp)

    # Factorial value for every cloud
    number_of_clouds = []
    for element in only_clouds:
        number_of_clouds.append(len(element))

    number_of_clouds = torch.tensor(number_of_clouds)
    if subsets.is_cuda:
        number_of_clouds = number_of_clouds.cuda()

    logBinomialCoefficient = log_binomial_coefficient(n=number_of_clouds, k=subsets)

    # calculating the error bound of the clouds based on logBinomialCoefficient
    prError = []
    for i, (subset, clouds, binomial) in enumerate(
        zip(subsets, only_clouds, logBinomialCoefficient)
    ):
        clouds = torch.sort(clouds)[
            0
        ]  # when you work with high precition float, to make operation in different order can give similar but diffetent values

        combination = torch.topk(batch[i, 0, clouds], subset, largest=True)[1]
        combination = clouds[combination]

        sumErrorCombination = torch.sum(torch.log(batch[i, 0, combination]))

        maskComplement = torch.tensor(
            [cloud not in combination for cloud in clouds], dtype=torch.bool
        )
        complementIndex = clouds[maskComplement]

        sumErrorComplement = torch.sum(torch.log(1 - batch[i, 0, complementIndex]))

        prError.append(sumErrorCombination + sumErrorComplement + binomial)

    return torch.stack(prError)