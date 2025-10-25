import numpy as np
import neurokit2 as nk


def shannon_entropy(x: np.array) -> float:
    """
    Calculate the Shannon entropy of a discrete random variable.

    Parameters:
        x (np.array): A 1D numpy array of discrete values.

    Returns:
        float: The Shannon entropy H(X) in nats.
    """
    e, _ = nk.entropy_shannon(x, base=np.e)
    return e


def joint_entropy(x: np.array, y: np.array) -> float:
    """
    Calculate the joint entropy H(X, Y) of two discrete random variables.

    Parameters:
        x (np.array): A 1D numpy array of discrete values for the first variable.
        y (np.array): A 1D numpy array of discrete values for the second variable.

    Returns:
        float: The joint entropy H(X, Y) in nats.
    """
    je, _ = nk.entropy_shannon_joint(
        x=x,
        y=y,
        base=np.e,
    )
    return je


def conditional_entropy(x: np.array, y: np.array) -> float:
    """
    Calculate the conditional entropy H(X|Y) of two discrete random variables.

    Parameters:
        x (np.array): A 1D numpy array of discrete values for the first variable.
        y (np.array): A 1D numpy array of discrete values for the second variable.

    Returns:
        float: The conditional entropy H(X|Y) in nats.
    """
    # Calculate joint entropy H(X, Y)
    je_xy = joint_entropy(x, y)

    # Calculate entropy H(Y)
    e_y = shannon_entropy(y)

    # Conditional entropy H(X|Y) = H(X, Y) - H(Y)
    ce = je_xy - e_y

    return ce
