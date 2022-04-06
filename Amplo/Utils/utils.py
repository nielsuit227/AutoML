__all__ = ['get_model', 'hist_search']


def get_model(model_str, **args):
    # Import here to prevent ImportError (due to circular import)
    from Amplo.AutoML.Modeller import Modeller
    try:
        model = [mod for mod in Modeller(**args).return_models()
                 if type(mod).__name__ == model_str]
        return model[0]
    except IndexError:
        raise IndexError('Model not found.')


def hist_search(array, value):
    low = 0
    high = len(array) - 1
    while low < high:
        middle = low + (high - low) // 2
        if value > array[middle + 1]:
            low = middle + 1
        elif value < array[middle]:
            high = middle
        else:
            return middle
    return -1
