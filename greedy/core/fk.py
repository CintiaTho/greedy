"""Provides a fractional knapsack implementation."""
from collections import namedtuple
from typing import Dict


data = namedtuple("data", "value, metric")


class Label(str):
    pass


def fractional_knapsack(
        items: Dict[Label, data],
        goal: float,
) -> Dict[Label, float]:
    """Calculates weight for each item to maximize into goal value.

    Args:
        items: mapping of an item label to its data.
        goal: goal value to achieve.

    Returns:
        A mapping of item label to its contribution weight.

    Example:
    >>> weights = fractional_knapsack(
    ...     items={
    ...         'Almond': data(value=440.0, metric=5.0),
    ...         'AlmondFloor': data(value=882.0, metric=9.0),
    ...         'BrazilianNut': data(value=801.0, metric=9.0),
    ...         'CashewNut': data(value=567.0, metric=7.0),
    ...         'ChestnutMix': data(value=495.0, metric=5.0),
    ...         'Corn': data(value=599.0, metric=10.0),
    ...         'Cranberry': data(value=497.0, metric=7.0),
    ...         'Date': data(value=400.0, metric=5.0),
    ...         'Pistachio': data(value=378.0, metric=2.0),
    ...         'Provolone': data(value=716.0, metric=4.0),
    ...         'PumpkinSeed': data(value=350.0, metric=5.0),
    ...         'Quinoa': data(value=367.2, metric=8.0),
    ...         'Raisin': data(value=894.0, metric=6.0),
    ...     },
    ...     goal=20.,
    ... )
    >>> for item in sorted(weights.items()):
    ...     print(" ".join([str(x) for x in item]))
    Almond 0.0
    AlmondFloor 0.3333333333333333
    BrazilianNut 0.0
    CashewNut 0.0
    ChestnutMix 1.0
    Corn 0.0
    Cranberry 0.0
    Date 0.0
    Pistachio 1.0
    Provolone 1.0
    PumpkinSeed 0.0
    Quinoa 0.0
    Raisin 1.0
    """

    weights = {label: 0. for label in items}
    ratio = {
        label: items[label].value/items[label].metric
        for label in items
    }
    from_max_ratio_sequence = sorted(
        ratio, key=lambda label: ratio[label], reverse=True
    )

    for candidate in from_max_ratio_sequence:
        if items[candidate].metric > goal:
            weights[candidate] = goal / items[candidate].metric
            goal = 0.
        else:
            weights[candidate] = 1.
            goal -= items[candidate].metric

    return weights


def main():

    try:
        n = int(input("Number of items: "))
        goal = float(input("Goal value: "))
        items = {}
        for i in range(n):
            label_, value_, metric_, *_ = input(
                "Provide label, value and metric for item {0}: ".format(i)
            ).split()
            value = float(value_)
            metric = float(metric_)
            items[Label(label_)] = data(value, metric)

        weight = fractional_knapsack(items, goal)

        for label in sorted(weight):
            print("{0}: {1}".format(label, weight[label]))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
