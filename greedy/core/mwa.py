"""Provides minimum weight arborescences from digraph."""
from pathlib import Path
from random import sample
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
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


class Arc:
    """Arc representation in a digraph.

    Attributes:
        outgoing: node from which the arc is coming from.
        incoming: node into which the arc is entering to.
        cost: integer value to represent cost of the arc.
    """

    def __init__(self, *, outgoing: Any, incoming: Any, cost: int) -> None:

        self._outgoing = (
            outgoing
            if isinstance(outgoing, Node)
            else Node(outgoing)
        )
        self._incoming = (
            incoming
            if isinstance(incoming, Node)
            else Node(incoming)
        )
        self._cost = int(cost)

    def __repr__(self) -> str:

        return (
            "Arc(outgoing={0}, incoming={1}, cost={2})"
            .format(repr(self.outgoing), repr(self.incoming), repr(self.cost))
        )

    def __eq__(self, other: Any) -> bool:

        if not other:
            return False
        if not isinstance(other, Arc):
            return False

        return all([
            self.outgoing == other.outgoing,
            self.incoming == other.incoming,
            self.cost == other.cost,
        ])

    def __hash__(self) -> int:

        return hash((self.outgoing, self.incoming, self.cost))

    @property
    def outgoing(self) -> Node:
        """Outgoing node getter."""

        return self._outgoing

    @property
    def incoming(self) -> Node:
        """Incoming node getter."""

        return self._incoming

    @property
    def cost(self) -> int:
        """Cost getter."""

        return self._cost

    @cost.setter
    def cost(self, value: int) -> int:
        """Cost setter."""

        value = int(value)

        if value < 0:
            raise ValueError("Arc new cost cannot be negative")

        self._cost = value

        return self._cost


class Digraph:
    """Directed graph representation.

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

        self.nodes = set()
        self.arcs = []

        for node in nodes:
            self.add_node(node)

        for raw_arc in arcs:
            arc = raw_arc
            if not isinstance(raw_arc, Arc):
                outgoing_, incoming_, cost_, *_ = raw_arc
                arc = Arc(outgoing=outgoing_, incoming=incoming_, cost=cost_)
            self.add_arc(arc)

    def __repr__(self) -> str:

        return "Digraph(nodes={0}, arcs={1})".format(
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
        if not isinstance(other, Digraph):
            return False

        return self.nodes == other.nodes and self.arcs == other.arcs

    def add_node(self, s: Node) -> None:
        """Properly add a node into digraph."""

        s = s if isinstance(s, Node) else Node(s)
        self.nodes.update({s})

    def add_arc(self, arc: Arc) -> None:
        """Properly add an arc into digraph, including nodes if needed."""

        self.arcs.append(arc)
        for node in (arc.outgoing, arc.incoming):
            self.add_node(node)

    def validate_arc(self, arc: Arc) -> bool:
        """Validate if a given arc uses digraph nodes."""
        return (
            # Nodes check
            {arc.outgoing, arc.incoming}.issubset(set(self.nodes))
            # Cost check
            and arc.cost >= 0
        )

    def outgoing_arcs(
            self,
            s: Node,
            *,
            arcs: Optional[Iterable[Arc]] = None
    ) -> Tuple[Arc, ...]:
        """Retrieves arcs outgoing from parameter node.

        Args:
            s: current node to be analyzed.
            arcs: optional parameter to use other arcs
                  instead of stored ones.

        Returns:
            Tuple containing arcs outgoing from node.
        """

        if arcs is None:
            arcs = self.arcs[:]

        outgoing = []

        for arc in arcs:
            if not self.validate_arc(arc):
                continue

            if s == arc.outgoing:
                outgoing.append(arc)

        return tuple(outgoing)

    def incoming_arcs(
            self,
            s: Node,
            *,
            arcs: Optional[Iterable[Arc]] = None
    ) -> Tuple[Arc, ...]:
        """Retrieves arcs incoming into parameter node.

        Args:
            s: current node to be analyzed.
            arcs: optional parameter to use other arcs
                  instead of stored ones.

        Returns:
            Tuple containing arcs incoming into node.
        """

        if arcs is None:
            arcs = self.arcs[:]

        incoming = []

        for arc in arcs:
            if not self.validate_arc(arc):
                continue

            if s == arc.incoming:
                incoming.append(arc)

        return tuple(incoming)

    def arcs_min_cost(
            self,
            rhoX: Iterable[Arc]
    ) -> int:
        """Retrieves minimum cost among selected arcs in rhoX.

        Args:
            rhoX: tuple of arcs which enter into a strong component.

        Returns:
            Integer value of minimum cost among selected arcs in rhoX.
        """

        rhoX = [arc for arc in rhoX if self.validate_arc(arc)]

        if not rhoX:
            return 0

        return min(arc.cost for arc in rhoX)

    def __entering_set(
            self,
            X: FrozenSet[Node],
            *,
            arcs: Optional[Iterable[Arc]] = None
    ) -> Tuple[Arc, ...]:
        """Retrieve all arcs which enters strong component X.

        Entering arcs are arcs which outgoing nodes does not belong to
        strong component passed as parameter whilst incoming nodes
        does belong to the same strong component.

        Args:
            X: strong component, as a set of nodes, to be analyzed.
            arcs: optional parameter to use other arcs
                  instead of stored ones.

        Returns:
            A tuple of arcs which enter into a strong component.
        """

        if arcs is None:
            arcs = self.arcs[:]

        rhoX = []

        for node in X:
            for arc in self.incoming_arcs(node, arcs=arcs):
                # Node in X is skipped
                if arc.outgoing in X:
                    continue

                rhoX.append(arc)

        return tuple(rhoX)

    def __dfs_visit(
            self,
            s: Node,
            order: List[Node],
            visited: Set[Node],
            *,
            arcs: Optional[Iterable[Arc]] = None
    ) -> None:
        """Visits unvisited nodes using zero cost arcs.

        Auxiliary recursive function to visits only unvisited nodes
        using zero cost arcs.

        Args:
            s: current node to be analyzed.
            order: list to store visit order of nodes.
            visited: set to store which node was already visited.
            arcs: optional parameter to use other arcs
                  instead of stored ones.
        """

        if arcs is None:
            arcs = self.arcs[:]

        visited.update({s})

        for arc in self.outgoing_arcs(s, arcs=arcs):

            # This algorithm consider only zero cost arcs
            if arc.incoming in visited or arc.cost != 0:
                continue

            # Recursively visit nodes which fulfill criteria
            self.__dfs_visit(arc.incoming, order, visited, arcs=arcs)

        order.append(s)

    def __dfs(
            self,
            *,
            arcs: Optional[Iterable[Arc]] = None
    ) -> Tuple[Node, ...]:
        """Calculates a node sequence using zero cost arcs path.

        Uses a depth-first strategy to visit nodes in a digraph.

        Args:
            arcs: optional parameter to use other arcs
                  instead of stored ones.

        Returns:
            A tuple to store visit order of nodes.

        Example:

        >>> path = Digraph._Digraph__dfs(
        ...     Digraph(
        ...         nodes={0, 1, 2, 3, 4},
        ...         arcs=(
        ...             Arc(outgoing=0, incoming=1, cost=5),
        ...             Arc(outgoing=0, incoming=2, cost=0),
        ...             Arc(outgoing=1, incoming=3, cost=0),
        ...             Arc(outgoing=3, incoming=4, cost=0),
        ...             Arc(outgoing=4, incoming=1, cost=0),
        ...         ),
        ...     )
        ... )
        ...
        >>> path in [
        ...     (Node("2"), Node("4"), Node("3"), Node("1"), Node("0")),
        ...     (Node("4"), Node("3"), Node("1"), Node("2"), Node("0")),
        ...     (Node("1"), Node("4"), Node("3"), Node("2"), Node("0")),
        ...     (Node("3"), Node("1"), Node("4"), Node("2"), Node("0")),
        ...     (Node('2'), Node('1'), Node('4'), Node('3'), Node('0')),
        ...     (Node("2"), Node("3"), Node("1"), Node("4"), Node("0")),
        ...     (Node("2"), Node("0"), Node("4"), Node("3"), Node("1")),
        ...     (Node("2"), Node("0"), Node("1"), Node("4"), Node("3")),
        ...     (Node("2"), Node("0"), Node("3"), Node("1"), Node("4")),
        ... ]
        ...
        True
        >>>
        """

        if arcs is None:
            arcs = self.arcs[:]

        order: List[Node] = []
        visited: Set[Node] = set()

        for s in self:
            # As this uses a depth-first strategy to visit,
            # if a node was already visited, no need to recalculate.
            if s in visited:
                continue

            self.__dfs_visit(s, order, visited, arcs=arcs)

        return tuple(order)

    def __reverse_dfs(
            self,
            order: Tuple[Node, ...],
            *,
            arcs: Optional[Iterable[Arc]] = None
    ) -> FrozenSet[FrozenSet[Node]]:
        """Calculates strong components of a digraph using a node sequence.

        Args:
            order: a sequence of nodes to be analyzed in reverse order.
            arcs: optional parameter to use other arcs
                  instead of stored ones.

        Returns:
            A frozen set containing strong components as
            frozen sets of nodes.

        Example:

        >>> strong_components = Digraph._Digraph__reverse_dfs(
        ...     Digraph(
        ...         nodes={0, 1, 2, 3, 4},
        ...         arcs=(
        ...             Arc(outgoing=0, incoming=1, cost=5),
        ...             Arc(outgoing=0, incoming=2, cost=0),
        ...             Arc(outgoing=1, incoming=3, cost=0),
        ...             Arc(outgoing=3, incoming=4, cost=0),
        ...             Arc(outgoing=4, incoming=1, cost=0),
        ...         )
        ...     ),
        ...     (Node("2"), Node("0"), Node("4"), Node("3"), Node("1"))
        ... )
        ...
        >>> assert strong_components == frozenset({
        ...     frozenset({Node(0)}),
        ...     frozenset({Node(2)}),
        ...     frozenset({Node(1), Node(3), Node(4)}),
        ... })
        ...
        >>>
        """

        def reverse_visit(
                self,
                r: Node,
                s: Node,
                zero_cost_components: Dict[Node, Union[Node, None]],
                strong: Set[Node],
                arcs: Iterable[Arc]
        ) -> None:
            """Internal function to mark node components.

            Auxiliary recursive function to visits only
            unvisited nodes using zero cost arcs based on
            previously calculated node sequence.

            Args:
                r: root node for current analyzed component.
                s: current node to be analyzed.
                zero_cost_components: store which component
                                      a node belongs to.
                strong: current strong component being analyzed.
                arcs: arcs to be considered for analyzed digraph.
            """

            # Set current node as same component as r node
            zero_cost_components[s] = r
            strong.update({s})

            for arc in self.incoming_arcs(s, arcs=arcs):

                # This algorithm consider only unvisited nodes
                #  using zero cost arcs
                if (
                        zero_cost_components[arc.outgoing] is not None or
                        arc.cost != 0
                ):
                    continue

                # Recursively visit nodes which fulfill criteria
                zero_cost_components[arc.outgoing] = r
                reverse_visit(
                    self, r, arc.outgoing, zero_cost_components, strong, arcs
                )

        if arcs is None:
            arcs = self.arcs[:]

        zero_cost_components: Dict[Node, Union[Node, None]] = {
            s: None for s in self
        }
        strong_components: List[Set[Node]] = []

        for s in reversed(order):
            # As this uses the result of a depth-first strategy,
            # if a node was already marked as part of a component,
            # no need to recalculate.
            if zero_cost_components[s] is not None:
                continue

            # New strong component found
            strong: Set[Node] = set()
            reverse_visit(self, s, s, zero_cost_components, strong, arcs)
            strong_components.append(strong)

        return frozenset(
            frozenset(component)
            for component in strong_components
        )

    def kosaraju(
            self,
            *,
            arcs: Optional[Iterable[Arc]] = None
    ) -> FrozenSet[FrozenSet[Node]]:
        """Calculates strong components of a digraph using Kosaraju.

        Args:
            arcs: optional parameter to use other arcs
                  instead of stored ones.

        Returns:
            A frozen set of strong components, as frozen set of nodes,
            of current digraph.

        Example:

        >>> strong_components = Digraph(
        ...     nodes={0, 1, 2, 3, 4},
        ...     arcs=(
        ...         Arc(outgoing=0, incoming=1, cost=5),
        ...         Arc(outgoing=0, incoming=2, cost=0),
        ...         Arc(outgoing=1, incoming=3, cost=0),
        ...         Arc(outgoing=3, incoming=4, cost=0),
        ...         Arc(outgoing=4, incoming=1, cost=0),
        ...     ),
        ... ).kosaraju()
        >>> assert strong_components == frozenset({
        ...     frozenset({Node(0)}),
        ...     frozenset({Node(2)}),
        ...     frozenset({Node(1), Node(3), Node(4)}),
        ... })
        >>>
        """

        if arcs is None:
            arcs = self.arcs[:]

        order = self.__dfs(arcs=arcs)
        return self.__reverse_dfs(order, arcs=arcs)

    def minimum_weight_arborescence(
            self,
            s: Node,
    ) -> Union[Tuple[Node, ...], None]:
        """Calculate a minimum weight arborescence if one exists.

        Args:
            s: node to be considered as root of arborescence.

        Returns:
            A tuple of a minimum weight arborescence path if one exists.
            None, otherwise.

        Raises:
            ValueError: node not in digraph.
        """

        s = s if isinstance(s, Node) else Node(s)

        if s not in self:
            raise ValueError("Node {0} not in digraph {1}".format(s, self))

        previous_strong_components = self.kosaraju()
        working_arcs: Iterable[Arc] = self.arcs[:]

        def reduce_cost(arc):
            nonlocal minimum_entering_arc_cost

            return Arc(
                outgoing=arc.outgoing,
                incoming=arc.incoming,
                cost=(
                    arc.cost-minimum_entering_arc_cost
                    if arc in rho_X
                    else arc.cost
                )
            )

        fully_reduced = False
        while not fully_reduced:

            fully_reduced = True

            for X in previous_strong_components:
                if s in X:
                    continue

                rho_X = self.__entering_set(X, arcs=working_arcs)
                minimum_entering_arc_cost = self.arcs_min_cost(rho_X)

                if minimum_entering_arc_cost:
                    fully_reduced = False
                    working_arcs = tuple(map(reduce_cost, working_arcs))

            # Recalculate strong components after reducing entering arcs' cost
            if not fully_reduced:
                previous_strong_components = self.kosaraju(arcs=working_arcs)

        reversed_path: List[Node] = []
        visited: Set[Node] = set()

        self.__dfs_visit(s, reversed_path, visited, arcs=working_arcs)

        # If a calculated path from a given root has not
        # the same number of nodes as the digraph itself,
        # there is no minimum weight arborescence.
        if len(reversed_path) != len(self):
            return None

        return tuple(reversed(reversed_path))

    def minimum_weight_arborescence_roots(self):
        """Calculates roots for minimum weight arborescence in digraph.

        Returns:
            A tuple of minimum weight arborescence paths as tuples.

        Example:

        >>> roots_tuple = Digraph(
        ...     nodes=["A", "B", "C", "D", "E", "F"],
        ...     arcs=(
        ...         Arc(outgoing="A", incoming="B", cost=0),
        ...         Arc(outgoing="A", incoming="C", cost=0),
        ...         Arc(outgoing="A", incoming="E", cost=0),
        ...         Arc(outgoing="B", incoming="D", cost=0),
        ...         Arc(outgoing="C", incoming="D", cost=2),
        ...         Arc(outgoing="C", incoming="F", cost=0),
        ...         Arc(outgoing="E", incoming="F", cost=3),
        ...    ),
        ... ).minimum_weight_arborescence_roots()
        ...
        >>> roots_tuple in [
        ...     ((Node('A'), Node('B'), Node('D'), Node('C'), Node('F'), Node('E')),),
        ...     ((Node('A'), Node('B'), Node('D'), Node('E'), Node('C'), Node('F')),),
        ...     ((Node('A'), Node('C'), Node('F'), Node('B'), Node('D'), Node('E')),),
        ...     ((Node('A'), Node('C'), Node('F'), Node('E'), Node('B'), Node('D')),),
        ...     ((Node('A'), Node('E'), Node('B'), Node('D'), Node('C'), Node('F')),),
        ...     ((Node('A'), Node('E'), Node('C'), Node('F'), Node('B'), Node('D')),),
        ... ]
        ...
        True
        """

        found = []

        for s in sorted(self):
            arborescence = self.minimum_weight_arborescence(s)
            if arborescence:
                found.append(arborescence)

        return tuple(found)

    def to_pdf(
            self,
            file_path: Path,
            arborescence: Optional[Tuple[Node, ...]] = None
    ) -> None:
        """Generate a PDF file of the digraph.

        Args:
            file_path: file system path to store PDF file.
            arborescence: optional parameter to highlight
                          a minimum weight arborescence in digraph.
        """

        try:
            import graphviz
        except ModuleNotFoundError:
            return

        start: Union[Node, None] = None
        highlight_arcs: List[Arc] = []

        if arborescence:
            start = arborescence[0]
            dfs_stack = [start]
            remaining = list(arborescence[1:])

            while len(highlight_arcs) < len(self) - 1:
                current = dfs_stack.pop()
                next_ = remaining[0]

                # Find arcs from current to next_
                candidate_arcs = tuple(
                    arc
                    for arc in self.outgoing_arcs(current)
                    if arc.incoming == next_
                )

                if not candidate_arcs:
                    # No arc found, so keep depth-first visit
                    # by discarding current from visit stack
                    continue

                remaining = remaining[1:]

                # Find minimum cost of those arcs
                minimum_cost = min(arc.cost for arc in candidate_arcs)
                highlight_arcs.append(
                    Arc(outgoing=current, incoming=next_, cost=minimum_cost)
                )

                # Prepares stack for next iteration
                dfs_stack.append(current)
                dfs_stack.append(next_)

        dot_digraph = graphviz.Digraph(
            comment="{0} {1}".format(repr(self), arborescence),
        )

        for node in self.nodes:

            if node is not None and node == start:
                shape = 'doublecircle'
            else:
                shape = 'circle'

            dot_digraph.node(
                node.value,
                node.value,
                shape=shape,
            )

        arcs = highlight_arcs if highlight_arcs else self.arcs

        for arc in arcs:

            dot_digraph.edge(
                tail_name=arc.outgoing.value,
                head_name=arc.incoming.value,
                label=str(arc.cost),
                style='solid',
            )

        dot_digraph.render(
            filename=str(file_path),
            view=False,
            cleanup=True,
            format='pdf',
            quiet=True,
        )


def data_input() -> Digraph:
    """Interactively input digraph data and returns one if valid."""

    digraph = Digraph()

    nodes, arcs = [
        int(k)
        for k in input("Enter number of nodes and arcs: ").split()
    ]

    for i in range(nodes):
        digraph.add_node(
            Node(
                input(
                    "Provide node {0} label: ".format(i)
                )
            )
        )

    for i in range(arcs):
        outgoing_, incoming_, cost_, *_ = input(
            "Provide outgoing node, incoming node and cost for arc {0}: "
            .format(i)
        ).split()
        digraph.add_arc(
            Arc(outgoing=outgoing_, incoming=incoming_, cost=int(cost_))
        )

    return digraph


def result_output(
        digraph: Digraph,
        result: Tuple[Tuple[Node]],
        base_dir: Optional[Path] = None,
) -> None:
    """Display results in human readable fashion.

    Args:
        digraph: digraph used to calculate.
        results: a sequence of calculated minimum weight arborescences.
        base_dir: directory to store PDF files.
    """

    if base_dir is None:
        base_dir = Path()

    digraph.to_pdf(file_path=(base_dir / "original"))

    print()
    print("Calculating output of {0}:".format(digraph))
    print()

    if result:
        print(
            "  Found {0} minimum weight arborescence root{1}:"
            .format(
                "a" if len(result) == 1 else len(result),
                "" if len(result) == 1 else "s",
            )
        )
        for path in result:
            root = path[0].value
            print("   - Root: {0}".format(root))
            print("     Path: {0}".format(', '.join([n.value for n in path])))
            digraph.to_pdf(
                file_path=(base_dir / "root_{0}".format(root)),
                arborescence=path,
            )
    else:
        print("  No valid root for arborescence found")


def main() -> None:
    """Interactive use of minimum weight arborescences implementation."""

    try:
        original = data_input()

        result = original.minimum_weight_arborescence_roots()

        result_output(original, result)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
