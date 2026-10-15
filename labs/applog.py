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

# ESC is built at runtime so this source file stays plain ASCII.
ESC = chr(27)
RESET, BOLD = ESC + "[0m", ESC + "[1m"
BLUE, GREEN, YELLOW, RED, CYAN = (
    ESC + "[34m",
    ESC + "[32m",
    ESC + "[33m",
    ESC + "[31m",
    ESC + "[36m",
)

# PASS sits between INFO (20) and WARNING (30).
PASS_LEVEL = 25
logging.addLevelName(PASS_LEVEL, "PASS")


class ColorFormatter(logging.Formatter):
    """Color only the level name; message text stays uncolored."""

    COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: CYAN,
        PASS_LEVEL: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
    }

    def format(self, record):
        """Format a record with a colored level name, then restore it."""
        # Remember the plain name first; we mutate the shared record below.
        original = record.levelname
        color = self.COLORS.get(record.levelno, "")
        record.levelname = f"{BOLD}{color}{original:<5}{RESET}"
        try:
            return super().format(record)
        finally:
            # Restore: the file handler shares this record and must stay plain.
            record.levelname = original


def get_logger(name):
    """Return a logger with .ok() (PASS), .warn_() and .fail() helpers."""
    log = logging.getLogger(name)
    # DEBUG at the logger lets each handler decide what to show; clearing
    # handlers makes repeated calls (and re-imports) idempotent.
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = "%(asctime)s [%(levelname)s] %(message)s"

    console = logging.StreamHandler(sys.stdout)
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
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

    # Attach the house vocabulary (PASS/WARN/FAIL) as methods so callers
    # never touch numeric levels directly.
    log.ok = lambda msg, *a: log.log(PASS_LEVEL, msg, *a)
    log.warn_ = log.warning  # trailing underscore avoids shadowing Logger.warn
    log.fail = log.error
    return log
