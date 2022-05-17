from typing import Any
from typing import List
from typing import Callable
from typing import Optional

import time


INTERVAL_TYPE = ("steps", "time")


def set(a: Any, b: Any) -> Any:
    return a if a is not None else b


class Action:
    def __init__(
        self,
        fn: Callable,
        fn_kwargs: dict,
        max_index: int,
        interval: float,
        interval_type: str,
        counter: int = 1,
        last_time: Optional[float] = None,
        buffer: List[Any] = [],
        use_latest: bool = False,
        save_updates: bool = True,
        clear_upates: bool = False
    ) -> None:
        assert interval_type in INTERVAL_TYPE, (
            f"The `t_type` has to be in {INTERVAL_TYPE}! "
            f"But {INTERVAL_TYPE} was provided!") 
        
        self.fn = fn
        self.fn_kwargs = fn_kwargs
        self.max_index = max_index
        self.counter = counter
        self.interval = interval
        self.interval_type = interval_type
        self.last_time = last_time or time.time()
        self.buffer = buffer
        self.use_latest = use_latest
        self.save_updates = save_updates
        self.clear_buffer = clear_upates

    def __call__(
        self, 
        index: Optional[int] = None,
        update: Optional[Any] = None,
        use_latest: Optional[bool] = None,
        save_updates: Optional[bool] = None,
        clear_buffer: Optional[bool] = None,
        **kwargs
    ) -> None:
        use_latest = set(use_latest, self.use_latest)
        save_updates = set(save_updates, self.save_updates)
        clear_buffer = set(clear_buffer, self.clear_buffer)

        print(self)

        if save_updates:
            self.buffer.append(update)

        now = time.time()
        if self.do_call(index, now):
            if not use_latest:
                update = self.buffer

            print("counter", self.counter)
            print({
                    **self.fn_kwargs,
                    **kwargs
                })

            self.fn(
                update=update,
                index=index,
                counter=self.counter,
                **{
                    **self.fn_kwargs,
                    **kwargs
                }
            )

            self.last_index = index
            self.last_time = now
            
            if clear_buffer:
                self.buffer = []
            
            self.counter = 0
        self.counter += 1

    def do_call(self, index: int, now: float) -> bool:
        max_reached = index == self.max_index
        
        if self.interval_type == "time":
            return not (now - self.last_time < self.interval) or max_reached
        return not (self.counter < self.interval) or max_reached 