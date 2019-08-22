"""Provides minimum spanning forest using Prim algorithm."""
from functools import total_ordering
from heapq import heappop, heappush
from pathlib import Path
from random import choice, sample
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
)


class Node(str):
    """Node representation.

    Attributes:
        value: str value of the node.
    """

    def __new__(cls, *args, **kwargs):
        return str.__new__(cls, *args, **kwargs)

    def __repr__(self) -> str:

        return '{0}({1})'.format(type(self).__name__, super().__repr__())

    @property
    def value(self) -> str:
        """Node value getter."""
        return super().__repr__()


@total_ordering
class Arc:
    """Arc representation in a graph.

    Attributes:
        nodes: two different nodes of the arc.
        cost: float value to represent cost of the arc.
    """

    def __init__(self, *, nodes: Iterable[Node], cost: float) -> None:

        nodes_: Set[Node] = set()
        for n in nodes:
            nodes_.update(
                {n if isinstance(n, Node) else Node(n)}
            )

        if len(nodes_) != 2:
            raise ValueError(
                "An arc must have exactly two nodes"
            )

        self._nodes: FrozenSet[Node] = frozenset(nodes_)

        self._cost = float(cost)

    def __repr__(self) -> str:

        return (
            "Arc(nodes={0}, cost={1})"
            .format(repr(self.nodes), repr(self.cost))
        )

    def __eq__(self, other: Any) -> bool:

        if not other:
            return False
        if not isinstance(other, Arc):
            return False

        return all([
            self.nodes == other.nodes,
            self.cost == other.cost,
        ])

    def __lt__(self, other: Any) -> bool:

        if not isinstance(other, Arc):
            raise TypeError("Another arc expected")

        if self.__eq__(other):
            return False

        if self.cost >= other.cost:
            return False

        return True

    def __hash__(self) -> int:

        return hash((self.nodes, self.cost))

    @property
    def nodes(self) -> FrozenSet[Node]:
        """Nodes getter."""

        return self._nodes

    @property
    def cost(self) -> float:
        """Cost getter."""

        return self._cost

    @cost.setter
    def cost(self, value: float) -> float:
        """Cost setter."""

        self._cost = float(value)

        return self._cost


class Graph:
    """Graph representation.

    Attributes:
        nodes: set of nodes in digraph.
        arcs: list of arcs in digraph.
    """
    nodes: Set[Node]
    arcs: List[Arc]

    def __init__(
            self,
            *,
            nodes: Optional[Iterable[Any]] = None,
            arcs: Optional[Iterable[Any]] = None,
    ) -> None:

        if nodes is None:
            nodes = set()
        if arcs is None:
            arcs = []

        self.nodes: Set[Node] = set()
        self.arcs: List[Arc] = []
        self.arcs_by_node: Dict[Node, Set[Arc]] = {}

        for node in nodes:
            self.add_node(node)

        for raw_arc in arcs:
            arc = raw_arc
            if not isinstance(raw_arc, Arc):
                one_, other_, cost_, *_ = raw_arc
                arc = Arc(nodes={Node(one_), Node(other_)}, cost=cost_)
            self.add_arc(arc)

    def __repr__(self) -> str:

        return "Graph(nodes={0}, arcs={1})".format(
            repr(sorted(self.nodes)), repr(self.arcs),
        )

    def __len__(self) -> int:

        return len(self.nodes)

    def __iter__(self) -> Iterable[Node]:
        """Randomized order iterable."""

        return iter(sample(self.nodes, len(self.nodes)))

    def __eq__(self, other: Any) -> bool:

        if not other:
            return False
        if not isinstance(other, Graph):
            return False

        return self.nodes == other.nodes and self.arcs == other.arcs

    def add_node(self, s: Node) -> None:
        """Properly add a node into digraph."""

        s = s if isinstance(s, Node) else Node(s)
        self.nodes.update({s})
        self.arcs_by_node.setdefault(s, set())

    def add_arc(self, arc: Arc) -> None:
        """Properly add an arc into digraph, including nodes if needed."""

        self.arcs.append(arc)
        self.nodes.update(arc.nodes)
        for node in arc.nodes:
            self.arcs_by_node.setdefault(node, set()).update({arc})

    def validate_arc(self, arc: Arc) -> bool:
        """Validate if a given arc uses graph nodes."""
        return arc.nodes.issubset(self.nodes)

    def prim(self) -> FrozenSet[Arc]:
        """Calculates a minimal spanning forest using Prim.

        Returns:
            A frozen set of arcs composing a minimal spanning forest.

        Example:

        >>> prim = Graph(
        ...     nodes={
        ...         'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII',
        ...     },
        ...     arcs=(
        ...         Arc(nodes={'I', 'II'}, cost=1),
        ...         Arc(nodes={'I', 'III'}, cost=5),
        ...         Arc(nodes={'I', 'IV'}, cost=3),
        ...         Arc(nodes={'II', 'III'}, cost=10),
        ...         Arc(nodes={'II', 'V'}, cost=8),
        ...         Arc(nodes={'III', 'V'}, cost=13),
        ...         Arc(nodes={'III', 'VI'}, cost=7),
        ...         Arc(nodes={'IV', 'VI'}, cost=-2),
        ...         Arc(nodes={'IV', 'VIII'}, cost=-5),
        ...         Arc(nodes={'V', 'VI'}, cost=6),
        ...         Arc(nodes={'V', 'VII'}, cost=4),
        ...         Arc(nodes={'V', 'VIII'}, cost=15),
        ...         Arc(nodes={'VI', 'VIII'}, cost=12),
        ...         Arc(nodes={'VII', 'VIII'}, cost=11),
        ...     ),
        ... ).prim()
        >>> prim == frozenset({
        ...     Arc(nodes={'I', 'II'}, cost=1),
        ...     Arc(nodes={'I', 'III'}, cost=5),
        ...     Arc(nodes={'I', 'IV'}, cost=3),
        ...     Arc(nodes={'IV', 'VI'}, cost=-2),
        ...     Arc(nodes={'IV', 'VIII'}, cost=-5),
        ...     Arc(nodes={'V', 'VI'}, cost=6),
        ...     Arc(nodes={'V', 'VII'}, cost=4),
        ... })
        True
        >>>
        """
        working_nodes: Set[Node] = set()
        msf_arcs: Set[Arc] = set()

        min_heap: List[Arc] = []

        while not min_heap:
            a_node = choice(list(set(self.nodes).difference(working_nodes)))
            working_nodes.update({a_node})
            for arc in self.arcs_by_node[a_node]:
                heappush(min_heap, arc)

        while min_heap:
            minimal_cost_arc = heappop(min_heap)
            if minimal_cost_arc.nodes.issubset(working_nodes):
                continue

            current_node = (
                set(minimal_cost_arc.nodes).difference(working_nodes).pop()
            )
            working_nodes.update(minimal_cost_arc.nodes)
            msf_arcs.update({minimal_cost_arc})

            # If a minimum spanning tree for whole graph was found,
            # finish processing
            if len(msf_arcs) == len(self.nodes) - 1:
                break

            arcs_to_be_inserted = (
                self.arcs_by_node[current_node].difference(set(min_heap))
            )
            for arc in arcs_to_be_inserted:
                heappush(min_heap, arc)

        return frozenset(msf_arcs)

    def to_pdf(
            self,
            file_path: Path,
            highlight_arcs: Optional[FrozenSet[Arc]] = None
    ) -> None:
        """Generate a PDF file of the graph.

        Args:
            file_path: file system path to store PDF file.
            highlight_arcs: optional parameter to highlight
                            selected arcs in graph.
        """

        try:
            import graphviz
        except ModuleNotFoundError:
            return

        if highlight_arcs is None:
            highlight_arcs = frozenset()

        dot_graph = graphviz.Graph(
            comment="{0} {1}".format(repr(self), highlight_arcs),
        )

        for node in self.nodes:
            dot_graph.node(
                node.value,
                node.value,
                shape='circle',
            )

        for arc in self.arcs:

            style = 'solid'
            if highlight_arcs and arc not in highlight_arcs:
                style = 'dotted'

            sorted_ = sorted(arc.nodes)

            dot_graph.edge(
                tail_name=sorted_[0].value,
                head_name=sorted_[-1].value,
                label=str(arc.cost),
                style=style,
            )

        dot_graph.render(
            filename=str(file_path),
            view=False,
            cleanup=True,
            format='pdf',
            quiet=True,
        )


def data_input() -> Graph:
    """Interactively input graph data and returns one if valid."""

    graph = Graph()

    nodes, arcs = [
        int(k)
        for k in input("Enter number of nodes and arcs: ").split()
    ]

    for i in range(nodes):
        graph.add_node(
            Node(
                input(
                    "Provide node {0} label: ".format(i)
                )
            )
        )

    for i in range(arcs):
        one_, other_, cost_, *_ = input(
            "Provide two nodes and cost for arc {0}: ".format(i)
        ).split()
        graph.add_arc(
            Arc(nodes={Node(one_), Node(other_)}, cost=float(cost_))
        )

    return graph


def result_output(
        graph: Graph,
        result: FrozenSet[Arc],
        base_dir: Optional[Path] = None,
) -> None:
    """Display results in human readable fashion.

    Args:
        graph: graph used to calculate.
        results: arcs of a minimum spanning forest.
        base_dir: directory to store PDF files.
    """

    if base_dir is None:
        base_dir = Path()

    graph.to_pdf(file_path=(base_dir / "original"))

    print()
    print("Calculating output of {0}:".format(graph))
    print()

    print(
        "  Found a minimal spanning forest."
    )
    print()
    print(
        "  Nodes:"
    )
    for node in sorted(graph.nodes):
        print("     - {0}".format(node))
    print()
    print(
        "  Arcs:"
    )
    arc_tuples = sorted([
        (*sorted(arc.nodes), arc.cost)
        for arc in result
    ])
    for arc_tuple in arc_tuples:
        print("     {0} to {1} (cost {2})".format(*arc_tuple))
    graph.to_pdf(
        file_path=(base_dir / "msf"),
        highlight_arcs=result,
    )


def main() -> None:
    """Interactive use of minimum weight arborescences implementation."""

    try:
        original = data_input()

        result_output(original, result=original.prim())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
