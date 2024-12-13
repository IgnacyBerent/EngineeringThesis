import numpy as np
from sklearn.neighbors import KernelDensity

def calculate_probabilities(data, bins):
    hist, edges = np.histogramdd(data, bins=bins, density=True)
    probabilities = hist / np.sum(hist)
    return probabilities, edges

def find_bin(edges, values):
    bin_indices = []
    for edge, value in zip(edges, values):
        bin_indices.append(np.digitize(value, edge) - 1)
    return tuple(bin_indices)

def transfer_entropy(X, Y, k=1, l=1, bins=10):
    """
    Calculate Transfer Entropy from X to Y.

    Parameters:
    X: np.ndarray, time series for X
    Y: np.ndarray, time series for Y
    k: int, history length for Y
    l: int, history length for X
    bins: int, number of bins for discretization

    Returns:
    float, Transfer Entropy T_{X -> Y}
    """
    # Prepare lagged vectors
    N = len(Y) - max(k, l)
    y_future = Y[max(k, l):]
    y_past = np.array([Y[i:i + k] for i in range(N)])
    x_past = np.array([X[i:i + l] for i in range(N)])

    # Joint probabilities
    joint_data = np.hstack([y_future[:, None], y_past, x_past])
    joint_prob, edges = calculate_probabilities(joint_data, bins=[bins] * joint_data.shape[1])

    # Conditional probabilities
    y_past_x_past_data = np.hstack([y_past, x_past])
    y_past_x_past_prob, _ = calculate_probabilities(y_past_x_past_data, bins=[bins] * y_past_x_past_data.shape[1])

    y_past_data = y_past
    y_past_prob, _ = calculate_probabilities(y_past_data, bins=[bins] * y_past_data.shape[1])

    # Transfer Entropy
    TE = 0
    for i in range(len(joint_data)):
        joint_bin = find_bin(edges, joint_data[i])
        y_past_x_past_bin = find_bin(edges[:-1], y_past_x_past_data[i])
        y_past_bin = find_bin(edges[:-2], y_past_data[i])

        # Ensure indices are within bounds
        if (all(j < len(joint_prob) for j in joint_bin) and
            all(j < len(y_past_x_past_prob) for j in y_past_x_past_bin) and
            all(j < len(y_past_prob) for j in y_past_bin)):

            P_joint = joint_prob[joint_bin]
            P_y_past_x_past = y_past_x_past_prob[y_past_x_past_bin]
            P_y_past = y_past_prob[y_past_bin]

            if P_joint > 0 and P_y_past_x_past > 0 and P_y_past > 0:
                TE += P_joint * np.log(P_joint / (P_y_past_x_past * P_y_past))

    return TE

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(1000)
    Y = 0.5 * X + np.random.rand(1000) * 0.5  # Y influenced by X

    TE = transfer_entropy(X, Y, k=1, l=2, bins=20)
    print(f"Transfer Entropy T(X -> Y): {TE}")
