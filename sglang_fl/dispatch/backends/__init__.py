# Backend abstract base class.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Backend(ABC):
    """
    Abstract base class for operator backends.

    Each backend provides implementations for a set of operators.
    Backends should implement is_available() to indicate whether
    the backend can be used in the current environment.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name string."""
        pass

    @property
    def vendor(self) -> Optional[str]:
        """Vendor name (for VENDOR type backends)."""
        return None
