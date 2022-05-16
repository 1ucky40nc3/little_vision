from typing import Any
from typing import Tuple
from typing import Callable
from typing import Optional
from typing import OrderedDict

import time


INTERVAL_TYPE = ("steps", "time")


def set(a: Any, b: Any) -> Any:
    return a if a is not None else b


class Action:
    def __init__(
        self,
        fn: Callable,
        fn_kwargs: dict,
        interval: float,
        interval_type: str,
        max_index: float,
        last_index: float = 0.,
        last_time: Optional[float] = None,
        updates: OrderedDict = OrderedDict(),
        use_latest: bool = False,
        save_updates: bool = True,
        clear_upates: bool = False
    ) -> None:
        assert interval_type in INTERVAL_TYPE, (
            f"The `t_type` has to be in {INTERVAL_TYPE}! "
            f"But {INTERVAL_TYPE} was provided!") 
        
        self.fn = fn
        self.fn_kwargs = fn_kwargs
        self.interval = interval
        self.interval_type = interval_type
        self.max_index = max_index
        self.last_index = last_index
        self.last_time = last_time or time.time()
        self.updates = updates
        self.use_latest = use_latest
        self.save_updates = save_updates
        self.clear_updates = clear_upates

    def __call__(
        self, 
        index: Optional[int] = None,
        update: Optional[Any] = None,
        use_latest: Optional[bool] = None,
        save_updates: Optional[bool] = None,
        clear_updates: Optional[bool] = None,
        reset_index: Optional[bool] = False,
        **kwargs
    ) -> None:
        now = time.time()
        self.updates[index] = update

        if self.do_call(index, now):
            if use_latest is None:
                use_latest = self.use_latest
            if not use_latest:
                update = list(self.updates.values())
            
            eta = (now - self.last_time) * (
                self.max_index - index)

            self.fn(
                update=update,
                index=index,
                eta=eta,
                **{
                    **self.fn_kwargs,
                    **kwargs
                }
            )

            self.last_index = index
            self.last_time = now

            clear_updates = set(
                clear_updates, 
                self.clear_updates)
            if clear_updates:
                self.updates = OrderedDict()
            if reset_index:
                self.last_index = 0.

        save_updates = set(
            save_updates, 
            self.save_updates)
        if not save_updates:
            self.updates = OrderedDict()

    def do_call(self, index: int, now: float) -> bool:
        curr = now if self.interval_type == "time" else index
        last = self.last_time if self.interval_type == "time" else self.last_index

        return curr - last >= self.interval or index == self.max_index