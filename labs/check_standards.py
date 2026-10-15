#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/check_standards.py

Purpose
-------
Audits every Python script in labs/ against the house standards so compliance
is a command, not a memory exercise. For each file it verifies: shebang, banner
docstring (Project/Author/File + Purpose/Security/Usage/Exit Codes sections),
dual-sink logging (applog or Stream+File handlers), docstrings on every
function and class, no bare print(), no hard-coded secrets, and a healthy
comment ratio. Hard rules FAIL; softer style rules WARN.

Security
--------
Read-only: it parses source with ast and never imports or executes the files
it audits. It prints findings, never file contents beyond the offending line
number.

Usage
-----
    python3 labs/check_standards.py                 # audit labs/*.py
    python3 labs/check_standards.py labs/lab02_entanglement.py

Log files
---------
logs/check_standards/check_standards_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Every file passed (warnings allowed).
1   At least one hard rule failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import argparse
import ast
import glob
import os
import re
import sys

from applog import get_logger

log = get_logger("check_standards")

# Required banner content. Fields are matched literally (including the aligned
# colon) so the banner stays visually uniform across files.
BANNER_FIELDS = ("Project :", "Author  :", "File    :")
SECTIONS = ("Purpose", "Security", "Usage", "Exit Codes")
# Heuristic only: catches a quoted literal assigned to a name such as password
# or token. It cannot see secrets built at runtime, which is the safe pattern.
SECRET_WORDS = "password|secret|token|api_key"
SECRET_RE = re.compile(rf"({SECRET_WORDS})\s*=\s*[\"'][^\"']+[\"']", re.IGNORECASE)
MIN_COMMENT_RATIO = 0.08  # comment lines per code line; below this = WARN
MAX_LINE = 88  # soft limit; long lines are WARN, not FAIL


def audit(path):
    """Return (fails, warns) lists of human-readable findings for one file."""
    # Hard rules go to fails (exit 1); style rules go to warns (exit 0).
    fails, warns = [], []
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    lines = src.split("\n")

    # Parse first: if the file does not even compile, nothing else matters.
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"syntax error line {exc.lineno}: {exc.msg}"], warns

    # The shebang makes the script directly executable on macOS and Linux.
    if not lines[0].startswith("#!/usr/bin/env python3"):
        fails.append("line 1 must be '#!/usr/bin/env python3'")

    # clean=False keeps indentation so the section underline regex below works.
    doc = ast.get_docstring(tree, clean=False) or ""
    for field in BANNER_FIELDS:
        if field not in doc:
            fails.append(f"banner missing field '{field.strip()}'")
    for sec in SECTIONS:
        if not re.search(rf"^{sec}\n-+", doc, re.MULTILINE):
            fails.append(f"banner missing section '{sec}'")
    if "=" * 79 not in doc:
        warns.append("banner is not ruled with '=' x 79")
    rel = os.path.relpath(path)
    if f"File    : {rel}" not in doc:
        warns.append(f"banner File field does not match path '{rel}'")

    # Logging: reuse applog, or prove both sinks are wired up locally.
    is_logger = os.path.basename(path) == "applog.py"
    uses_applog = "from applog import" in src
    own_sinks = "StreamHandler" in src and "FileHandler" in src
    if not (is_logger or uses_applog or own_sinks):
        fails.append("no dual-sink logging (import applog or add Stream+File)")

    # One AST walk covers two rules: docstrings on defs, and no bare print().
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not ast.get_docstring(node):
                fails.append(f"line {node.lineno}: '{node.name}' lacks a docstring")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            warns.append(f"line {node.lineno}: use logging instead of print()")

    # Line-level rules. Report line numbers only, never the line itself, so a
    # matched secret is not echoed into the log file.
    for i, text in enumerate(lines, 1):
        if SECRET_RE.search(text):
            fails.append(f"line {i}: possible hard-coded secret")
        if len(text) > MAX_LINE:
            warns.append(f"line {i}: {len(text)} chars (> {MAX_LINE})")

    # Comment density is a cheap proxy for "explains why"; it cannot judge
    # quality, so it only warns.
    comments = sum(1 for t in lines if t.strip().startswith("#"))
    code = sum(1 for t in lines if t.strip() and not t.strip().startswith("#"))
    ratio = comments / max(code, 1)
    if ratio < MIN_COMMENT_RATIO:
        warns.append(f"comment ratio {ratio:.2f} < {MIN_COMMENT_RATIO}")
    return fails, warns


def main(argv=None):
    """Audit each file, log findings, and return the process exit code."""
    ap = argparse.ArgumentParser(description="Audit scripts against standards")
    ap.add_argument("files", nargs="*", help="default: labs/*.py")
    args = ap.parse_args(argv)
    files = args.files or sorted(glob.glob("labs/*.py"))
    if not files:
        ap.error("no Python files found; run from the project root")

    # Count failing FILES (not findings) so the summary reads naturally.
    failed = 0
    for path in files:
        fails, warns = audit(path)
        for w in warns:
            log.warn_("%s: %s", path, w)
        for f in fails:
            log.fail("%s: %s", path, f)
        if fails:
            failed += 1
        else:
            log.ok("%s meets standards (%d warning(s))", path, len(warns))
    if failed:
        log.fail("%d of %d file(s) failed", failed, len(files))
        return 1
    log.ok("All %d file(s) meet standards", len(files))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
