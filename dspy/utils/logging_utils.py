import logging
import logging.config
import sys

LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"


class DSPyLoggingStream:
    """
    A Python stream for use with event logging APIs throughout DSPy (`eprint()`,
    `logger.info()`, etc.). This stream wraps `sys.stderr`, forwarding `write()` and
    `flush()` calls to the stream referred to by `sys.stderr` at the time of the call.
    It also provides capabilities for disabling the stream to silence event logs.
    """

    def __init__(self) -> None:
        self._enabled: bool = True

    def write(self, text: str) -> None:
        """Write *text* to ``sys.stderr`` if the stream is enabled."""
        if self._enabled:
            sys.stderr.write(text)

    def flush(self) -> None:
        """Flush ``sys.stderr`` if the stream is enabled."""
        if self._enabled:
            sys.stderr.flush()

    @property
    def enabled(self) -> bool:
        """Whether this logging stream is currently active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value


DSPY_LOGGING_STREAM = DSPyLoggingStream()


def disable_logging() -> None:
    """
    Disables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), silencing all subsequent event logs.
    """
    DSPY_LOGGING_STREAM.enabled = False


def enable_logging() -> None:
    """
    Enables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), emitting all subsequent event logs. This
    reverses the effects of `disable_logging()`.
    """
    DSPY_LOGGING_STREAM.enabled = True


def configure_dspy_loggers(root_module_name: str) -> None:
    """Configure a DSPy-specific logging handler for the given root module.

    Attaches a :class:`logging.StreamHandler` that writes to
    :data:`DSPY_LOGGING_STREAM` (and therefore to ``sys.stderr``) to the
    logger identified by *root_module_name*. If a handler with the same name
    already exists on the logger it is replaced, preventing duplicate log
    entries on repeated calls (e.g., during hot-reload or tests).

    Args:
        root_module_name: The dotted Python module name used as the logger
            name, typically ``"dspy"``.
    """
    formatter = logging.Formatter(fmt=LOGGING_LINE_FORMAT, datefmt=LOGGING_DATETIME_FORMAT)

    dspy_handler_name = "dspy_handler"
    handler = logging.StreamHandler(stream=DSPY_LOGGING_STREAM)
    handler.setFormatter(formatter)
    handler.set_name(dspy_handler_name)

    logger = logging.getLogger(root_module_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for existing_handler in logger.handlers[:]:
        if getattr(existing_handler, "name", None) == dspy_handler_name:
            logger.removeHandler(existing_handler)

    logger.addHandler(handler)
