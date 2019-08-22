"""Provides a k-means algorithm implementation."""
from typing import Callable, List

import numpy as np


def distance_(x: np.ndarray, y: np.ndarray) -> float:
    """Arbitrary distance function."""
    z = x - y
    return np.inner(z, z)


def kmeans(
        input_: np.ndarray,
        k: int,
        steps: int = 0,
        distance: Callable[[np.ndarray, np.ndarray], float] = distance_,
) -> np.ndarray:
    """Randomly calculates k clusters and provides their mean points.

    Args:
        input_: array containing all data points.
        k: number of clusters to be calculated.
        steps: number of iteration steps to be done at most.
        distance: function to calculate distance between two points.

    Returns:
        K points which represent mean point for each cluster.
    """

    if steps < 1:
        steps = 255

    data_count = len(input_)
    cluster_data = np.array([
        input_[i] for i in np.random.choice(data_count, k)
    ])
    found = False
    current_step = 0
    while not found and current_step < steps:
        closest_cluster = np.zeros(data_count)
        near_count = np.zeros(k)
        new_cluster_data = np.zeros(cluster_data.shape)

        for data_index in range(data_count):
            closest_cluster_index = np.argmin([
                distance(input_[data_index], cluster)
                for cluster in cluster_data
            ])
            closest_cluster[data_index] = closest_cluster_index
            near_count[closest_cluster_index] += 1
            new_cluster_data[closest_cluster_index] += input_[data_index]

        from_scratch = False
        cluster_index = 0
        while not from_scratch and cluster_index < k:
            if not near_count[cluster_index]:
                from_scratch = True
                continue
            new_cluster_data[cluster_index] /= near_count[cluster_index]
            cluster_index += 1

        if from_scratch:
            new_cluster_data = [
                input_[i] for i in np.random.choice(data_count, k)
            ]
            current_step += 1
            continue

        if np.array_equal(cluster_data, new_cluster_data):
            found = True
        else:
            cluster_data = new_cluster_data
            current_step += 1

    return cluster_data


def main() -> None:

    try:
        input_: List[np.ndarray] = []
        dimensions = 0

        k = int(input("K: "))
        steps = int(input("Enter maximum number of steps: "))
        n = int(input("Enter number of data points: "))
        for i in range(n):
            input_.append(np.array([
                float(v)
                for v in input(
                    "Enter input values for data point {0}: "
                    .format(i)
                ).split()
            ]))
            if i:
                if dimensions != len(input_[-1]):
                    raise ValueError(
                        "Number of dimensions among data points differ"
                    )
            else:
                dimensions = len(input_[0])

        means = kmeans(np.array(input_), k, steps)

        print("{0}-means:".format(k))
        for i in range(means.shape[0]):
            print("mean {0}: {1}".format(i, means[i]))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
