"""Provides k-center algorithm implementation."""
from typing import Dict, FrozenSet, Sequence, Set, Tuple, Union


INF = float('inf')


class Label(str):
    pass


def argmax(
        pi_: Dict[Label, float],
    ) -> Label:
    """Calculates label which value is greatest."""
    return max(pi_, key=lambda label: pi_[label])


def kcenter(
        labels: Sequence[Label],
        distances: Dict[FrozenSet[Label], float],
        k: int,
    ) -> Tuple[FrozenSet[Label], Dict[Label, Label]]:
    """Calculates K-centers for parameter data.

    Args:
        labels: sequence of labels.
        distances: dict of pair of labels and respective float values.
        k: value to determine k-center for input data.

    Return:
        A tuple containing a frozen set of chosen k-labels and
        a mapping of label to related chosen label as its center.

    Example:
    >>> chosen, center = kcenter(
    ...     labels=['Alfa', 'Bravo', 'Charlie', 'Delta', 'Epsilon'],
    ...     distances={
    ...         frozenset({'Alfa'}): 0,
    ...         frozenset({'Bravo'}): 0,
    ...         frozenset({'Charlie'}): 0,
    ...         frozenset({'Delta'}): 0,
    ...         frozenset({'Epsilon'}): 0,
    ...         frozenset({'Bravo', 'Alfa'}): 12.0,
    ...         frozenset({'Charlie', 'Alfa'}): 23.0,
    ...         frozenset({'Delta', 'Alfa'}): 34.0,
    ...         frozenset({'Epsilon', 'Alfa'}): 45.0,
    ...         frozenset({'Bravo', 'Charlie'}): 56.0,
    ...         frozenset({'Delta', 'Bravo'}): 67.0,
    ...         frozenset({'Bravo', 'Epsilon'}): 78.0,
    ...         frozenset({'Delta', 'Charlie'}): 89.0,
    ...         frozenset({'Epsilon', 'Charlie'}): 90.0,
    ...         frozenset({'Delta', 'Epsilon'}): 21.0,
    ...     },
    ...     k=3,
    ... )
    >>> sorted(chosen)
    ['Alfa', 'Charlie', 'Epsilon']
    >>> " ".join("{0}:{1}".format(*pair) for pair in sorted(center.items()))
    'Alfa:Alfa Bravo:Alfa Charlie:Charlie Delta:Epsilon Epsilon:Epsilon'
    """

    chosen_labels: Set[Label] = set()
    center_for_label: Dict[Label, Union[Label, None]] = {
        label: None for label in labels
    }
    pi_: Dict[Label, float] = {label: INF for label in labels}

    while len(chosen_labels) < k:
        chosen = argmax(pi_)
        chosen_labels.update({chosen})
        for other in labels:
            pair = frozenset({chosen, other})
            if pi_[other] > distances[pair]:
                pi_[other] = distances[pair]
                center_for_label[other] = chosen

    return frozenset(chosen_labels), center_for_label


def main():

    try:
        k = int(input("K: "))
        n = int(input("Number of labels: "))

        labels = []
        for index in range(n):
            labels.append(input("Label {0}: ".format(index)))

        distances = {frozenset({label}): 0 for label in labels}
        for index_a, label_a in enumerate(labels[:-1]):
            for label_b in labels[index_a+1:]:
                pair = frozenset({label_a, label_b})
                distances[pair] = float(
                    input(
                        "Distance of {0} and {1}: "
                        .format(label_a, label_b)
                    )
                )

        chosen_labels, center_for = kcenter(labels, distances, k)
        print(
            "Chosen {0}-centers: {1}"
            .format(k, ", ".join(sorted(chosen_labels)))
        )
        for label in sorted(center_for):
            print(
                "  Center for {0}: {1}"
                .format(label, center_for[label])
            )

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
