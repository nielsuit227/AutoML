import logging
import warnings


__all__ = ['get_model', 'hist_search']


def get_model(model_str, **kwargs):
    # Import here to prevent ImportError (due to circular import)
    from Amplo.AutoML.Modeller import Modeller
    try:
        model = [mod for mod in Modeller(**kwargs).return_models()
                 if type(mod).__name__ == model_str]
        return model[0]
    except IndexError:
        raise IndexError('Model not found.')


def hist_search(array, value):
    """
    Binary search that finds the index in order to fulfill
    ``array[index] <= value < array[index + 1]``

    Parameters
    ----------
    array : array of float
    value : float

    Returns
    -------
    int
        Bin index of the value
    """

    # Return -1 when no bin exists
    if value < array[0] or value >= array[-1]:
        logging.debug(f'No bin (index) found for value {value}. Array(Min: {array[0]}, Max: {array[-1]})')
        return -1

    # Initialize min and max bin index
    low = 0
    high = len(array) - 1

    # Bin search
    countdown = 30
    while countdown > 0:
        # Count down
        countdown -= 1

        # Set middle bin index
        middle = low + (high - low) // 2

        if low == middle == high - 1:  # stop criterion
            return middle

        if value < array[middle]:  # array[low] <= value < array[middle]
            high = middle
        elif value >= array[middle]:  # array[middle] <= value < array[high]
            low = middle

    warnings.warn(RuntimeWarning('Operation took too long. Returning -1 (no match).'))
    return -1
