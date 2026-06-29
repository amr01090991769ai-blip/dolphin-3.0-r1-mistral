"""Filesystem sandbox utilities.

All file-touching tools must resolve user-supplied paths through
``safe_path`` so the agent can never read or write outside the configured
workspace (defends against path-traversal like ``../../etc/passwd``).
"""
from __future__ import annotations

import os
from pathlib import Path


class SandboxError(Exception):
    pass


def ensure_workspace(workspace: str) -> Path:
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_path(workspace: str, user_path: str) -> Path:
    """Resolve *user_path* and guarantee it stays inside *workspace*."""
    root = ensure_workspace(workspace)
    # Treat absolute user paths as relative to the workspace root.
    candidate = (root / user_path.lstrip("/\\")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise SandboxError(
            f"Path '{user_path}' escapes the workspace sandbox and was blocked."
        )
    return candidate
