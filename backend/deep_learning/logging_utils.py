from __future__ import annotations

import logging
import os
import warnings


class LightningNoiseFilter(logging.Filter):
    """Suppress non-actionable Lightning promotional messages."""

    _BLOCKED_SUBSTRINGS = (
        "for seamless cloud logging",
        "for seamless cloud uploads",
        "try installing [litlogger]",
        "try installing [litmodels]",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage().lower()
        return not any(substring in msg for substring in self._BLOCKED_SUBSTRINGS)


def suppress_lightning_noise() -> None:
    """
    Reduce noisy Lightning advice/promotional logs while keeping real warnings/errors.
    """
    os.environ.setdefault("DISABLE_LIGHTNING_ADVICE", "1")

    for logger_name in (
        "lightning",
        "lightning.pytorch",
        "lightning.pytorch.utilities.rank_zero",
        "pytorch_lightning",
    ):
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.WARNING)
        if not any(isinstance(f, LightningNoiseFilter) for f in lg.filters):
            lg.addFilter(LightningNoiseFilter())

    warnings.filterwarnings(
        "ignore",
        message=r".*try installing \[litlogger\].*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*try installing \[litmodels\].*",
    )


def configure_cli_logging(level: int = logging.INFO) -> None:
    """Configure project CLI logs and reduce third-party INFO chatter."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for logger_name in (
        "lightning",
        "lightning.pytorch",
        "lightning.pytorch.utilities.rank_zero",
        "pytorch_lightning",
        "torch",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
