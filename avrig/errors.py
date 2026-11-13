"""Typed, user-facing failures. Each carries what happened, what to do, and an exit code."""
from __future__ import annotations


class RigError(Exception):
    code = 4  # runtime failure

    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.msg, self.hint = msg, hint


class ConfigError(RigError):
    code = 2


class BusyError(RigError):
    """A device or the instance lock is held by another process."""
    code = 3


class PreflightError(RigError):
    """Recording refused: at least one FAIL-level health issue."""
    code = 1

    def __init__(self, issues: list):
        self.issues = issues
        fails = [i for i in issues if i.level == "FAIL"]
        super().__init__(f"preflight failed: {len(fails)} blocking issue(s)",
                         "fix the issues above, or record anyway with --force")
