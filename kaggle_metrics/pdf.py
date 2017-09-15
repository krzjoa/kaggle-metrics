# Kaggle metrcis
# Krzysztof Joachimiak 2017

# Metrics for probability distribution function


def crps(y_true, y_pred):
    '''

    Continuous Ranked Probability Score

    Parameters
    ----------
    y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class predictions (0 or 1 values only)

    Returns
    ------
    score: float
        Continuous Ranked Probability Score score

    References
    ----------
    .. [1] https://www.kaggle.com/wiki/ContinuousRankedProbabilityScore
    .. [2] http://journals.ametsoc.org/doi/pdf/10.1175/1520-0434%282000%29015%3C0559%3ADOTCRP%3E2.0.CO%3B2

    '''