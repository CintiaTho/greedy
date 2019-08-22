"""Provides task schedulers."""
from typing import Any, Dict, Iterable, Optional, Tuple, Union


class Label(str):
    """Label class to clear intention."""
    pass


class TaskTimeInterval:
    """Representation of a time interval for task."""

    def __init__(self, *, start: Any, end: Any) -> None:

        start = float(start)
        end = float(end)

        if end <= start:
            raise ValueError("End must be greater than start")

        self._start = start
        self._end = end

    def __repr__(self) -> str:

        return (
            "TaskTimeInterval(start={0}, end={1})"
            .format(repr(self.start), repr(self.end))
        )

    def __eq__(self, other: Any) -> bool:

        if not other:
            return False
        if not isinstance(other, TaskTimeInterval):
            return False

        return all([
            self.start == other.start,
            self.end == other.end,
        ])

    def __hash__(self) -> int:

        return hash((self.start, self.end))

    @property
    def end(self) -> float:
        """End getter."""
        return self._end

    @property
    def start(self) -> float:
        """Start getter."""
        return self._start


class TaskScheduler:
    """Scheduler representation.

    Attributes:
        tasks: dict of labels for time interval tasks.
    """
    _tasks: Dict[Label, TaskTimeInterval]

    def __init__(
            self,
            *,
            tasks: Optional[
                Dict[Any, Union[Iterable[float], TaskTimeInterval]]
            ] = None,
    ) -> None:

        if tasks is None:
            tasks = {}

        self._tasks: Dict[Label, TaskTimeInterval] = {}

        for task_label in tasks:
            task_time_interval = tasks[task_label]
            if not isinstance(task_time_interval, TaskTimeInterval):
                start_, end_, *_ = task_time_interval
                task_time_interval = TaskTimeInterval(start=start_, end=end_)

            self.add_task(
                label=Label(task_label),
                time_interval=task_time_interval,
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
    def tasks(self) -> Dict[Label, TaskTimeInterval]:
        """Tasks getter."""
        return self._tasks

    def add_task(
            self,
            *,
            label: Label,
            time_interval: TaskTimeInterval
    ) -> None:
        """Properly add a task into scheduler."""

        if label in self.tasks:
            raise ValueError(
                "Already exists a task labeled {0}"
                .format(label)
            )

        self._tasks[label] = time_interval

    def schedule(self) -> Tuple[Label, ...]:
        """Calculates an order for tasks to be scheduled.

        Returns:
            Resulting tuple of task label according to scheduled order.

        Example:
        >>> TaskScheduler(
        ...     tasks={
        ...         "Alfa": TaskTimeInterval(start=1, end=6),
        ...         "Bravo": TaskTimeInterval(start=2, end=5),
        ...         "Charlie": TaskTimeInterval(start=1, end=4),
        ...         "Delta": TaskTimeInterval(start=4, end=5),
        ...         "Echo": TaskTimeInterval(start=2, end=3),
        ...         "Foxtrot": TaskTimeInterval(start=1, end=2),
        ...         "Golf": TaskTimeInterval(start=2, end=7),
        ...         "Hotel": TaskTimeInterval(start=3, end=4),
        ...         "India": TaskTimeInterval(start=6, end=7),
        ...         "Juliett": TaskTimeInterval(start=7, end=8),
        ...     }
        ... ).schedule()
        ...
        ('Foxtrot', 'Echo', 'Hotel', 'Delta', 'India', 'Juliett')
        >>>
        """

        ending_sooner = sorted(
            self.tasks,
            key=lambda task: self.tasks[task].end
        )

        if not ending_sooner:
            return ()

        task_order = [ending_sooner[0]]

        remaining = ending_sooner[1:]

        for tentative in remaining:
            if self.tasks[tentative].start >= self.tasks[task_order[-1]].end:
                task_order.append(tentative)

        return tuple(task_order)


def data_input() -> TaskScheduler:
    """Interactively input task data and returns a scheduler if valid."""

    scheduler = TaskScheduler()

    tasks = int(input("Enter number of tasks: "))

    for i in range(tasks):
        label_, start_, end_, *_ = input(
            "Provide label, start value and"
            " end value for task {0}: ".format(i)
        ).split()
        scheduler.add_task(
            label=Label(label_),
            time_interval=TaskTimeInterval(start=start_, end=end_),
        )

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
        for label in task_order:
            interval = scheduler.tasks[label]
            print(
                "    {0}: {1} to {2}"
                .format(label, interval.start, interval.end)
            )
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
