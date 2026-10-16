#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/exlib.py

Purpose
-------
Shared helper for the optional-exercise scripts (ex01 .. ex10). Every exercise
is logged in the same shape: the QUESTION, your PREDICTION (written before the
code runs), the MEASURED result, and a PASS/FAIL verdict comparing the two.
That keeps each exercise a real experiment instead of a printout.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    from exlib import Tally, base_parser
    t = Tally(log)
    t.ask("Exercise 1: ...")
    t.predict("the answer I expect")
    t.check(cond, "what was verified")
    return t.finish()

Log files
---------
None of its own; the importing script owns the log.

Exit Codes
----------
Library module; finish() returns 0 (all passed) or 1 (any failed).
===============================================================================
"""

import argparse

# Imported so a Tally built without a logger still logs to screen and file.
from applog import get_logger


def base_parser(description):
    """Argparse parser shared by all exercise scripts (seed is the only knob)."""
    # One common parser keeps --seed and --help identical across ten scripts,
    # so a user never has to relearn the interface per exercise.
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    return ap


class Tally:
    """Collects check results for one exercise script and prints the verdict."""

    def __init__(self, log=None):
        """Keep the script's logger and an empty result list."""
        # Scripts normally pass their own logger so all lines share one file;
        # the fallback keeps the class usable on its own.
        self.log = log or get_logger("exlib")
        self.results = []

    def ask(self, title):
        """Open a new exercise section in the log."""
        self.log.info("=== %s ===", title)

    def predict(self, text):
        """Record the prediction BEFORE the code computes the answer."""
        self.log.info("PREDICT  : %s", text)

    def note(self, fmt, *args):
        """Log a measured value or an explanation."""
        self.log.info("MEASURED : " + fmt, *args)

    def check(self, cond, label):
        """Log PASS/FAIL for one claim and remember it for the exit code."""
        # bool() strips NumPy booleans so later count(False) works reliably.
        (self.log.ok if cond else self.log.fail)("%s", label)
        self.results.append(bool(cond))
        return bool(cond)

    def finish(self):
        """Summarize and return the process exit code (0 = all passed)."""
        # Failures are counted, not raised, so one bad check never hides the rest.
        bad = self.results.count(False)
        if bad:
            self.log.fail("%d of %d checks failed", bad, len(self.results))
            return 1
        self.log.ok("All %d checks passed", len(self.results))
        return 0
