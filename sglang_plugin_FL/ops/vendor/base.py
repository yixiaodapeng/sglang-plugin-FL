"""VendorBackend abstract base class.

Each chip vendor implements a subclass with:
- name/vendor properties for identification
- is_available() for hardware detection
- operator methods (silu_and_mul, rms_norm, etc.)
"""

from abc import ABC, abstractmethod


class VendorBackend(ABC):
    """Abstract base for chip-vendor operator backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name, e.g. 'cuda', 'ascend', 'mlu'."""
        ...

    @property
    @abstractmethod
    def vendor(self) -> str:
        """Vendor identifier, e.g. 'nvidia', 'ascend', 'cambricon'."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the hardware for this backend is detected."""
        ...
