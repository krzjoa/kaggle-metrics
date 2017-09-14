# Kaggle metrcis
# Krzysztof Joachimiak 2017


class WrongShapesExpception(Exception):
    pass

def check_shapes(y_true, y_pred):

    # TODO: check 1-dim vs 2-dim

    if not cmp(y_true.shape, y_pred.shape):
        raise WrongShapesExpception("Array shapes arr inconsistent "
                                    "y_true: {}"
                                    "y_pred: {}".format(y_true.shape, y_pred.shape))




def confusion_binary(y_true, y_pred):
    # TODO: !!!!
    return None, None, None, None