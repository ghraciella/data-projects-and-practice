"""Functions used in preparing Guido's gorgeous lasagna.

Learn about Guido, the creator of the Python language:
https://en.wikipedia.org/wiki/Guido_van_Rossum

This is a module docstring, used to describe the functionality
of a module and its functions and/or classes.
"""



EXPECTED_BAKE_TIME = 40
PREPARATION_TIME_PER_LAYER = 2

def bake_time_remaining(elapsed_bake_time):
    """Calculate the bake time remaining.

    :param elapsed_bake_time: int - baking time already elapsed.
    :return: int - remaining bake time (in minutes) derived from 'EXPECTED_BAKE_TIME'.

    Function that takes the actual minutes the lasagna has been in the oven as
    an argument and returns how many minutes the lasagna still needs to bake
    based on the `EXPECTED_BAKE_TIME`.
    """

    if isinstance(elapsed_bake_time, int):
        time_left = abs(EXPECTED_BAKE_TIME - elapsed_bake_time)

    return time_left


def preparation_time_in_minutes(number_of_layers):
    """Calculate preparation time in minutes.

    :param number_of_layers: int - the number of layers to add to the lasagna.
    :return: int - preparation time (in minutes) to make the lasagna layers, derived from 'PREPARATION_TIME_PER_LAYER'.

    Function that takes tthe number of layers to add to the lasagna as
    an argument and returns how many minutes one would spend making them
    based on the `PREPARATION_TIME_PER_LAYER`.
    """


    if isinstance(number_of_layers, int):
        time_taken = abs(PREPARATION_TIME_PER_LAYER * number_of_layers)

    return time_taken


def elapsed_time_in_minutes(number_of_layers, elapsed_bake_time):
    """Calculate  total elapsed cooking time (prep + bake) in minutes.

    :param number_of_layers: int - the number of layers added to the lasagna.
    :param elapsed_bake_time: int - the number of minutes the lasagna has been baking in the oven.
    :return: int - total time elapsed (in minutes) preparing and cooking.

    Function that takes the number of layers added to the lasagna and the number of minutes 
    the lasagna has been baking in the oven as arguments and returns the total number of minutes 
    you've been cooking, or the sum of your preparation time and the time the lasagna has already 
    spent baking in the oven.
    """


    if (isinstance(number_of_layers, int) and isinstance(elapsed_bake_time, int)):
        preparation_and_bake_time = preparation_time_in_minutes(number_of_layers) + elapsed_bake_time 

    return preparation_and_bake_time