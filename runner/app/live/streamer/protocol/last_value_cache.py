import threading
from typing import Optional, TypeVar, Generic, Callable

T = TypeVar('T')

class LastValueCache(Generic[T]):
    """
    A thread-safe cache that stores the last value produced.
    Consumers can wait for a value to be available with a timeout.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._UNSET = object()
        self._value = self._UNSET

    def put(self, value: T) -> None:
        """Store a new value and notify all waiting consumers."""
        with self._lock:
            self._value = value
            self._condition.notify_all()

    def get(self, timeout: float = 30.0) -> Optional[T]:
        """
        Get the latest value, waiting if necessary.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            The latest value, or None if timeout occurs before a value is available
        """
        def value_is_set():
            return self._value is not self._UNSET

        with self._condition:
            # Wait for a value to be available
            if not value_is_set():
                success = self._condition.wait_for(value_is_set, timeout=timeout)
                if not success:
                    logging.warning(f"Timed out waiting for value (timeout={timeout}s)")
                    return None

            return self._value


