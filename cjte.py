import numpy as np
from scipy.linalg import lstsq

def calculate_cjte(xi, y, xc, model_order_range=(4, 16)):
    """
    Calculate the Conditional Joint Transfer Entropy (CJTE).

    Parameters:
    - xi: np.array, the input signal.
    - y: np.array, the output signal.
    - xc: np.array, the conditioning signal.
    - model_order_range: tuple, range of model orders to optimize Akaike Information Criterion (AIC).

    Returns:
    - cjte: float, the computed CJTE value.
    """
    def normalize(signal, epsilon=1e-8):
        """Normalize the signal by subtracting mean and dividing by standard deviation."""
        std = np.std(signal)
        if std < epsilon:
            return signal - np.mean(signal)  # If std is too small, just subtract the mean
        return (signal - np.mean(signal)) / std

    def compute_prediction_error(y, predictors):
        """Compute prediction error variance using linear regression."""
        predictors = np.column_stack([predictors, np.ones(len(predictors))])  # Add bias term
        coefficients, _, _, _ = lstsq(predictors, y)
        predictions = np.dot(predictors, coefficients)
        error = y - predictions
        return np.var(error)

    def optimize_model_order(y, predictors, model_order_range):
        """Find the optimal model order using AIC."""
        min_aic = float("inf")
        best_order = None
        for order in range(model_order_range[0], model_order_range[1] + 1):
            predictors_ordered = predictors[:len(y) - order]
            y_ordered = y[order:]
            predictors_shifted = np.column_stack([np.roll(predictors_ordered, -i, axis=0)[:-order] for i in range(order)])
            y_shifted = y_ordered[:len(y_ordered) - order]
            error_variance = compute_prediction_error(y_shifted, predictors_shifted)
            aic = len(y_shifted) * np.log(error_variance) + 2 * order
            if aic < min_aic:
                min_aic = aic
                best_order = order
        return best_order

    # Step 1: Normalize the signals
    xi, y, xc = map(normalize, [xi, y, xc])

    # Step 2: Form predictors for full and restricted universes
    predictors_full = np.column_stack([y[:-1], xi[:-1], xc[:-1]])
    predictors_restricted = np.column_stack([y[:-1], xc[:-1]])

    # Step 3: Optimize model order
    model_order = optimize_model_order(y[:-1], predictors_full, model_order_range)

    # Step 4: Fit models and compute prediction error variances
    predictors_full_shifted = np.column_stack([np.roll(predictors_full, -i, axis=0)[:-model_order] for i in range(model_order)])
    predictors_restricted_shifted = np.column_stack([np.roll(predictors_restricted, -i, axis=0)[:-model_order] for i in range(model_order)])
    y_shifted = y[model_order:][:len(predictors_full_shifted)]

    sigma_y_yxc = compute_prediction_error(y_shifted, predictors_restricted_shifted)
    sigma_y_yxixc = compute_prediction_error(y_shifted, predictors_full_shifted)

    # Step 5: Compute CJTE
    cjte = 0.5 * np.log(sigma_y_yxc / sigma_y_yxixc)

    return cjte
