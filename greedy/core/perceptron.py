"""Provides an implementation of Perceptron algorithm."""
from typing import List, Union

import numpy as np


def wrong(
        input_: np.ndarray,
        output: np.ndarray,
        weight: np.ndarray,
) -> Union[int, None]:

    for index, input_array in enumerate(input_):
        if output[index]*(np.inner(weight, input_array)) <= 0:
            return index

    return None


def perceptron(
        input_: np.ndarray,
        output: np.ndarray,
) -> np.ndarray:

    weight = np.zeros(input_.shape[-1])

    adjust_needed = True
    while adjust_needed:
        index = wrong(input_, output, weight)
        if index is None:
            adjust_needed = False
            continue
        weight += output[index]*input_[index]

    return weight


def main() -> None:

    try:
        input_: List[np.ndarray] = []
        output: List[float] = []

        n = int(input("Enter number of samples: "))
        for i in range(n):
            output_value, *input_value = [
                float(v)
                for v in input(
                    "Enter output value and input values for sample {0}: "
                    .format(i)
                ).split()
            ]
            output.append(output_value)
            input_.append(np.array([1, *input_value]))

        weight = perceptron(np.array(input_), np.array(output))

        print("weight:", weight, end="\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
