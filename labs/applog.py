#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/applog.py

Purpose
-------
Shared logging for every lab so each lab file stays focused on the physics.
Console output is colored by severity (only on a TTY with NO_COLOR unset);
the file sink is always plain text so logs stay greppable. Stdlib only.

Security
--------
Handles no secrets. Writes only plain-text logs under logs/<name>/.

Usage
-----
    from applog import get_logger
    log = get_logger("lab02")
    log.info("hello"); log.ok("checked"); log.warn_("careful"); log.fail("bad")

Log files
---------
logs/<name>/<name>_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
Library module; it never exits the process.
===============================================================================
"""

import logging
import os
import sys
from datetime import datetime
from typing import ClassVar, cast

# Build ANSI escapes at runtime to keep the source plain ASCII.
ESC = chr(27)
RESET = ESC + "[0m"
BOLD = ESC + "[1m"
DIM = ESC + "[2m"
GREEN = ESC + "[32m"
YELLOW = ESC + "[33m"
RED = ESC + "[31m"
CYAN = ESC + "[36m"

# PASS sits between INFO (20) and WARNING (30).
PASS_LEVEL = 25
logging.addLevelName(PASS_LEVEL, "PASS")


class ColorFormatter(logging.Formatter):
    """Color only the level name; message text stays uncolored."""

    COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: DIM,
        logging.INFO: CYAN,
        PASS_LEVEL: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record):
        """Format a record with a colored level name, then restore it."""
        # Remember the plain name first; we mutate the shared record below.
        original = record.levelname
        color = self.COLORS.get(record.levelno, "")
        if color:
            emphasis = "" if record.levelno == logging.DEBUG else BOLD
            record.levelname = f"{emphasis}{color}{original:<5}{RESET}"
        else:
            record.levelname = f"{original:<5}"
        try:
            return super().format(record)
        finally:
            # Restore: the file handler shares this record and must stay plain.
            record.levelname = original


class LabLogger(logging.Logger):
    """Logger carrying the house vocabulary (PASS/WARN/FAIL) as real methods.

    Declaring them on a subclass, rather than attaching lambdas at runtime,
    lets type checkers (Pylance/mypy) see .ok/.warn_/.fail on every caller.
    stacklevel=2 keeps %(funcName)s/%(lineno)d pointing at the caller.
    """

    def ok(self, msg, *args, **kwargs):
        """Log at PASS (25)."""
        kwargs.setdefault("stacklevel", 2)
        self.log(PASS_LEVEL, msg, *args, **kwargs)

    def warn_(self, msg, *args, **kwargs):
        """Log at WARNING; trailing underscore avoids shadowing Logger.warn."""
        kwargs.setdefault("stacklevel", 2)
        self.warning(msg, *args, **kwargs)

    def fail(self, msg, *args, **kwargs):
        """Log at ERROR."""
        kwargs.setdefault("stacklevel", 2)
        self.error(msg, *args, **kwargs)


def get_logger(name) -> LabLogger:
    """Return a logger with .ok() (PASS), .warn_() and .fail() helpers."""
    log = logging.getLogger(name)
    # Swap the class on this one logger instead of calling the global
    # logging.setLoggerClass, which would also change third-party loggers
    # (PennyLane, Qiskit). Safe because LabLogger adds methods, not state.
    if not isinstance(log, LabLogger):
        log.__class__ = LabLogger
    log = cast(LabLogger, log)
    # DEBUG at the logger lets each handler decide what to show; clearing
    # handlers makes repeated calls (and re-imports) idempotent.
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = "%(asctime)s [%(levelname)s] %(message)s"

    console = logging.StreamHandler(sys.stdout)
    color = (
        sys.stdout.isatty()
        and os.environ.get("TERM", "") != "dumb"
        and "NO_COLOR" not in os.environ
    )
    console.setFormatter(
        (ColorFormatter if color else logging.Formatter)(fmt, "%H:%M:%S")
    )
    log.addHandler(console)

    # Best-effort file sink: a read-only filesystem must not crash a lab.
    try:
        d = os.path.join("logs", name)
        os.makedirs(d, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(d, f"{name}_{stamp}.log")
        fh = logging.FileHandler(path)
        fh.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)
        log.info("Log file: %s", path)
    except OSError as exc:
        log.warning("File logging disabled (%s); console only.", exc)
    return log
