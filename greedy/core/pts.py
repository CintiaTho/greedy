"""Provides schedulers for tasks with penalty."""
from typing import (
    Any, Callable, Dict, FrozenSet, Iterable, Optional, Set, Tuple
)


class Label(str):
    """Label class to clear intention."""
    pass


class TaskWithPenalty:
    """Representation of a task with precedence."""

    def __init__(
            self,
            *,
            duration: float,
            penalty: Callable[[float], float],
            dependent_on: Optional[Iterable[Label]] = None,
    ) -> None:

        if dependent_on is None:
            dependent_on = []

        duration = float(duration)

        if duration <= 0.:
            raise ValueError("Duration must be positive")

        self._duration = duration
        self._dependent_on = frozenset(dependent_on)
        self._penalty = penalty

    def __repr__(self) -> str:

        return (
            "TaskWithDependency(duration={0}, dependent_on={1}, penalty={2})"
            .format(
                repr(self.duration),
                repr(self.dependent_on),
                repr(self.penalty)
            )
        )

    def __eq__(self, other: Any) -> bool:

        if not other:
            return False
        if not isinstance(other, TaskWithPenalty):
            return False

        return all([
            self.duration == other.duration,
            self.dependent_on == other.dependent_on,
            self.penalty == other.penalty,
        ])

    def __hash__(self) -> int:

        return hash((self.duration, self.dependent_on, self.penalty))

    @property
    def duration(self) -> float:
        """Duration getter."""
        return self._duration

    @property
    def dependent_on(self) -> FrozenSet:
        """Getter of frozen set of dependency."""
        return self._dependent_on

    @property
    def penalty(self) -> Callable[[float], float]:
        """Penalty function getter."""
        return self._penalty


class TaskScheduler:
    """Scheduler representation.

    Attributes:
        tasks: dict of labels mapping to dependent tasks with penalty.
    """
    _tasks: Dict[Label, TaskWithPenalty]

    def __init__(
            self,
            *,
            tasks: Optional[Dict[Label, TaskWithPenalty]] = None,
    ) -> None:

        if tasks is None:
            tasks = {}

        self._tasks: Dict[Label, TaskWithPenalty] = {}

        for task_label in tasks:
            task_with_penalty = tasks[task_label]
            if not isinstance(task_with_penalty, TaskWithPenalty):
                continue

            self.add_task(
                label=Label(task_label),
                task_with_penalty=task_with_penalty,
            )

    def __repr__(self) -> str:

        return "TaskScheduler(tasks={0})".format(
            repr(self.tasks),
        )

    def __len__(self) -> int:

        return len(self.tasks)

    def __eq__(self, other: Any) -> bool:

        if not other:
            return False
        if not isinstance(other, TaskScheduler):
            return False

        return self.tasks == other.tasks

    def __hash__(self) -> int:

        return hash(self.tasks)

    @property
    def tasks(self) -> Dict[Label, TaskWithPenalty]:
        """Tasks getter."""
        return self._tasks

    def add_task(
            self,
            *,
            label: Label,
            task_with_penalty: TaskWithPenalty
    ) -> None:
        """Properly add a task into scheduler."""

        if label in self.tasks:
            raise ValueError(
                "Already exists a task labeled {0}"
                .format(label)
            )

        self._tasks[label] = task_with_penalty

    def schedule(self) -> Tuple[Label, ...]:
        """Calculates an order for tasks to be scheduled.

        Returns:
            Resulting tuple of task label according to scheduled order.

        Example:
        >>> TaskScheduler(
        ...     tasks={
        ...         "Alpha": TaskWithPenalty(
        ...             duration=3,
        ...             dependent_on=[],
        ...             penalty=lambda x: x ** 2,
        ...         ),
        ...         "Beta": TaskWithPenalty(
        ...             duration=1,
        ...             dependent_on=["Alpha", "Gamma"],
        ...             penalty=lambda x: x ** 3,
        ...         ),
        ...         "Gamma": TaskWithPenalty(
        ...             duration=4,
        ...             dependent_on=[],
        ...             penalty=lambda x: x + 1,
        ...         ),
        ...         "Delta": TaskWithPenalty(
        ...             duration=2,
        ...             dependent_on=["Beta"],
        ...             penalty=lambda x: x,
        ...         ),
        ...         "Epsilon": TaskWithPenalty(
        ...             duration=9,
        ...             dependent_on=["Beta", "Delta", "Zeta"],
        ...             penalty=lambda x: 1,
        ...         ),
        ...         "Zeta": TaskWithPenalty(
        ...             duration=8,
        ...             dependent_on=["Gamma"],
        ...             penalty=lambda x: x + 2,
        ...         ),
        ...         "Eta": TaskWithPenalty(
        ...             duration=7,
        ...             dependent_on=["Zeta"],
        ...             penalty=lambda x: x**2 + 1,
        ...         ),
        ...     }
        ... ).schedule()
        ...
        ('Alpha', 'Gamma', 'Beta', 'Zeta', 'Eta', 'Delta', 'Epsilon')
        >>>
        """

        total_duration = sum(
            [self.tasks[task].duration for task in self.tasks]
        )

        reversed_task_order = []

        # Set requirement relations up
        requirement_for: Dict[Label, Set[Label]] = {
            label: set() for label in self.tasks
        }
        for task_label in self.tasks:
            for required_label in self.tasks[task_label].dependent_on:
                if required_label not in self.tasks:
                    raise ValueError(
                        "{0} depends on unregistered task {1}"
                        .format(task_label, required_label)
                    )
                requirement_for[required_label].update({task_label})

        least_dependency = sorted(
            requirement_for,
            key=lambda task: len(requirement_for[task])
        )

        if not least_dependency:
            return ()

        current_ending_time = total_duration

        sink = [
            task
            for task in requirement_for
            if not requirement_for[task]
        ]

        while current_ending_time:
            if not sink:
                raise ValueError("No acyclic graph from task dependency found")

            sink_enumeration = iter(enumerate(sink))
            chosen_index, minimal_penalty_label = next(sink_enumeration)
            minimal_penalty_task = self.tasks[minimal_penalty_label]
            for tentative_index, tentative_label in sink_enumeration:
                analyzed_task = self.tasks[tentative_label]
                if (
                        analyzed_task.penalty(current_ending_time) <
                        minimal_penalty_task.penalty(current_ending_time)
                ):
                    chosen_index = tentative_index
                    minimal_penalty_label = tentative_label
                    minimal_penalty_task = analyzed_task

            # Update data of tasks depending on selected task
            for required_task_label in minimal_penalty_task.dependent_on:
                requirement_for[required_task_label].remove(minimal_penalty_label)
                if not requirement_for[required_task_label]:
                    sink.append(required_task_label)

            # Do not consider this task again
            del requirement_for[minimal_penalty_label]
            sink[-1], sink[chosen_index] = sink[chosen_index], sink[-1]
            sink.pop()

            # Insert into selected tasks
            reversed_task_order.append(minimal_penalty_label)
            current_ending_time -= minimal_penalty_task.duration

        return tuple(reversed(reversed_task_order))


def _penalty_treatment(expression: str) -> Callable[[float], float]:
    """Parses expression considering that has at most one variable.

    Parses using Abstract Syntax Tree to analyze a textual expression.

    Args:
        expression: a textual expression of what is expected to be
                    an one float input variable and float return value.

    Raises:
        ValueError: expression does not pass triage validation.
    """

    from ast import parse, Expr, NodeVisitor

    class NameCollector(NodeVisitor):
        """NodeVisitor subclass to gather variable names."""
        names: Set[str]

        def __init__(self):
            self.names = set()
            super().__init__()

        def visit_Name(self, node):
            """Register variable names."""
            self.names.update({node.id})

    node = parse(expression)

    if len(node.body) > 1:
        raise ValueError(
            "'{0}' must be only one expression"
            .format(expression)
        )

    if not isinstance(node.body[0], Expr):
        raise ValueError(
            "'{0}' must be only one expression"
            .format(expression)
        )

    collector = NameCollector()
    collector.visit(node)

    if len(collector.names) > 1:
        raise ValueError("At most one variable is allowed")

    if len(collector.names) == 1:
        return eval(
            "lambda {0}: {1}".format(list(collector.names)[0], expression)
        )

    return eval("lambda _: {0}".format(expression))


def data_input() -> TaskScheduler:
    """Interactively input task data and returns a scheduler if valid."""

    scheduler = TaskScheduler()

    tasks = int(input("Enter number of tasks: "))

    for i in range(tasks):
        # Read data of a single task
        label_, duration_, *dependent_on_ = input(
            "Provide label, duration and list of precedence tasks'"
            " labels for task {0}: ".format(i)
        ).split()
        # Input mathematical expression to be set as penalty
        penalty_ = input(
            "Provide a penalty function for task {0}"
            " (ex.: x ** 2 - x + 8): ".format(i)
        )

        label = Label(label_)
        duration = float(duration_)
        dependent_on = set(
            Label(requirement)
            for requirement in dependent_on_
        )
        penalty = _penalty_treatment(penalty_)

        task = TaskWithPenalty(
            duration=duration,
            penalty=penalty,
            dependent_on=dependent_on,
        )
        scheduler.add_task(label=label, task_with_penalty=task)

    return scheduler


def result_output(
        scheduler: TaskScheduler,
) -> None:
    """Display results in human readable fashion.

    Args:
        scheduler: scheduler used to calculate.
    """

    print()
    print("Calculating output of {0}:".format(scheduler))
    print()

    task_order = scheduler.schedule()

    if task_order:
        print(
            "  Found {0} scheduled task{1}:"
            .format(
                "a" if len(task_order) == 1 else len(task_order),
                "" if len(task_order) == 1 else "s",
            )
        )
        current_initial = 0.
        for label in task_order:
            current_task = scheduler.tasks[label]
            next_initial = current_initial + current_task.duration
            print(
                "    {0}: from {1} to {2}"
                .format(label, current_initial, next_initial)
            )
            current_initial = next_initial
    else:
        print("  No available task found")


def main() -> None:
    """Interactive use of task scheduler implementation."""

    try:
        scheduler = data_input()

        result_output(scheduler=scheduler)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
