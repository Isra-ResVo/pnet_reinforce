from pnet_reinforce.generator.generator.initial_data import FAILURE_PROBABILITY
import numpy as np

# originally this fucntion has the element self as first
# arguemnt this was change because is a bad practice.
def optimisticPessimistic(selected_clouds_indices) -> np.ndarray:

    r"""
    Returs a numpy array with two elements:
    -   First element: contain the best probabilities of static CSP data,
    -   Second element: conatain the worst probabilities of  static CSP data.

    Final shape: (2,1,number of CSP) this keep consistency with the rest of code.
    
    The main idea behid this function is to provide a way to obtain in form of array
    the worst and better probality of one set of clouds. This can be used to graph
    and compare where the solution with respect to this these values.

    CSP = Cloud service provider
    """
    # Get the values the probabilities from a dict
    fail_prob = FAILURE_PROBABILITY

    if selected_clouds_indices is not None:
        arg_iter = selected_clouds_indices.cpu().numpy()
    else:
        arg_iter = fail_prob

    array = np.array([[] for _ in range(2)])

    for key in arg_iter:
        mini, maxi = fail_prob[key]
        array = np.concatenate((array, np.array([[mini], [maxi]])), axis=1)
    array = np.expand_dims(array, axis=1)
    array = np.sort(array, axis=-1)

    return array