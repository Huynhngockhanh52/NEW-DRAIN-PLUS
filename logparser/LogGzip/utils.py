from collections.abc import Sequence

import numpy as np
import scipy.stats


def NCD(c1: float, c2: float, c12: float) -> float:
    """
    Calculates Normalized Compression Distance (NCD).

    Arguments:
        c1 (float): The compressed length of the first object.
        c2 (float): The compressed length of the second object.
        c12 (float): The compressed length of the concatenation of the first
                     and second objects.

    Returns:
        float: The Normalized Compression Distance c1 and c2.

    Formula:
        NCD(c1, c2, c12) = (c12 - min(c1, c2)) / max(c1, c2)
    """

    distance = (c12 - min(c1, c2)) / max(c1, c2)
    return distance



def agg_by_concat_space(t1: str, t2: str) -> str:
    """
    Combines `t1` and `t2` with a space.

    Arguments:
        t1 (str): First item.
        t2 (str): Second item.

    Returns:
        str: `{t1} {t2}`
    """

    return t1 + " " + t2



#
#
def mean_confidence_interval(data: Sequence, confidence: float = 0.95) -> tuple:
    """
    Computes the mean confidence interval of `data` with `confidence`

    Arguments:
        data (Sequence): Data to compute a confidence interval over.
        confidence (float): Level to compute confidence.

    Returns:
        tuple: (Mean, quantile-error-size)
    """

    if isinstance(data, np.ndarray):
        array = data
    else:
        array = np.array(data, dtype=np.float32)

    n = array.shape[0]

    mean = np.mean(array)
    standard_error = scipy.stats.sem(array)
    quantile = scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, standard_error * quantile
