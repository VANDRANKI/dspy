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

    def __init__(self):
        self._enabled = True

    def write(self, text):
        if self._enabled:
            sys.stderr.write(text)

    def flush(self):
        if self._enabled:
            sys.stderr.flush()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value


DSPY_LOGGING_STREAM = DSPyLoggingStream()


def disable_logging():
    """
    Disables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), silencing all subsequent event logs.
    """
    DSPY_LOGGING_STREAM.enabled = False


def enable_logging():
    """
    Enables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), emitting all subsequent event logs. This
    reverses the effects of `disable_logging()`.
    """
    DSPY_LOGGING_STREAM.enabled = True


def configure_dspy_loggers(root_module_name: str) -> None:
    """Attach the DSPy stream handler to the named logger.

    Sets up a `logging.StreamHandler` (writing to `DSPY_LOGGING_STREAM`) on the
    logger identified by `root_module_name`, formatted with `LOGGING_LINE_FORMAT`.
    The logger's level is set to `INFO` and propagation is disabled so DSPy's
    formatting isn't duplicated by ancestor loggers. Safe to call more than once:
    any previously attached DSPy handler is removed before the new one is added.

    Args:
        root_module_name: Name of the logger to configure, typically `__name__`
            of the package's `__init__.py` (e.g. `"dspy"`).
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
