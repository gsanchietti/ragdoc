from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_STATE_DIR = Path(".ragdoc")
_STATE_FILE = _STATE_DIR / "state.json"


def _ensure_state_dir() -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict[str, Any]:
    _ensure_state_dir()
    if _STATE_FILE.exists():
        try:
            with _STATE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state: dict[str, Any]) -> None:
    _ensure_state_dir()
    tmp = _STATE_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, _STATE_FILE)


def set_http_meta(state: dict[str, Any], url: str, etag: str | None, last_mod: str | None) -> None:
    http = state.setdefault("http", {})
    http[url] = {"etag": etag, "last_modified": last_mod}


def get_http_meta(state: dict[str, Any], url: str) -> tuple[str | None, str | None]:
    meta = state.get("http", {}).get(url, {})
    return meta.get("etag"), meta.get("last_modified")


def set_git_meta(state: dict[str, Any], repo: str, commit: str) -> None:
    git = state.setdefault("git", {})
    git[repo] = {"commit": commit}


def get_git_meta(state: dict[str, Any], repo: str) -> str | None:
    return state.get("git", {}).get(repo, {}).get("commit")
