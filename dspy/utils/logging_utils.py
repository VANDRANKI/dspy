import logging
import logging.config
import sys
from typing import Optional

LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"


class DSPyLoggingStream:
    """
    A Python stream for use with event logging APIs throughout DSPy (`eprint()`,
    `logger.info()`, etc.). This stream wraps `sys.stderr`, forwarding `write()` and
    `flush()` calls to the stream referred to by `sys.stderr` at the time of the call.
    It also provides capabilities for disabling the stream to silence event logs.

    Example usage::

        import dspy
        dspy.disable_logging()  # Silence all DSPy log output
        # ... run your program ...
        dspy.enable_logging()   # Re-enable log output
    """

    def __init__(self) -> None:
        self._enabled: bool = True

    def write(self, text: str) -> None:
        """Write text to stderr if the stream is enabled.

        Args:
            text: The string to write to the underlying stderr stream.
        """
        if self._enabled:
            sys.stderr.write(text)

    def flush(self) -> None:
        """Flush the underlying stderr stream if the stream is enabled."""
        if self._enabled:
            sys.stderr.flush()

    @property
    def enabled(self) -> bool:
        """Whether this logging stream is currently enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable this logging stream.

        Args:
            value: Set to ``True`` to enable output, ``False`` to suppress it.
        """
        self._enabled = value


DSPY_LOGGING_STREAM = DSPyLoggingStream()


def disable_logging() -> None:
    """
    Disables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), silencing all subsequent event logs.

    Example::

        import dspy
        dspy.disable_logging()  # No more DSPy log messages
    """
    DSPY_LOGGING_STREAM.enabled = False


def enable_logging() -> None:
    """
    Enables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), emitting all subsequent event logs. This
    reverses the effects of `disable_logging()`.

    Example::

        import dspy
        dspy.disable_logging()
        # ... silent section ...
        dspy.enable_logging()  # Resume log output
    """
    DSPY_LOGGING_STREAM.enabled = True


def configure_dspy_loggers(root_module_name: str) -> None:
    """Configure DSPy loggers to write to the DSPy logging stream.

    Sets up a :class:`logging.StreamHandler` that forwards log records produced
    by ``root_module_name`` and all its child loggers to
    :data:`DSPY_LOGGING_STREAM`.  Any previously registered DSPy handler is
    replaced so that duplicate log lines are never emitted.

    Args:
        root_module_name: The top-level Python logger name to configure
            (e.g. ``"dspy"``).  All child loggers (``dspy.predict``,
            ``dspy.teleprompt``, etc.) inherit this configuration.

    Example::

        from dspy.utils.logging_utils import configure_dspy_loggers
        configure_dspy_loggers("dspy")
    """
    formatter = logging.Formatter(fmt=LOGGING_LINE_FORMAT, datefmt=LOGGING_DATETIME_FORMAT)

    dspy_handler_name: str = "dspy_handler"
    handler: logging.StreamHandler = logging.StreamHandler(stream=DSPY_LOGGING_STREAM)
    handler.setFormatter(formatter)
    handler.set_name(dspy_handler_name)

    logger: logging.Logger = logging.getLogger(root_module_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for existing_handler in logger.handlers[:]:
        if getattr(existing_handler, "name", None) == dspy_handler_name:
            logger.removeHandler(existing_handler)

    logger.addHandler(handler)
