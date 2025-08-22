from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .models import GitSource
from .state import get_git_meta, set_git_meta, load_state, save_state
from .utils import ensure_dir, safe_name

logger = logging.getLogger(__name__)


class GitRepoFetcher:
    def __init__(self) -> None:
        pass

    def fetch_one(self, src: GitSource, out_root: Path) -> Path:
        repo_dir = out_root / src.out_dir / safe_name(src.repo)
        ensure_dir(repo_dir.parent)

        if not (repo_dir / ".git").exists():
            logger.info("Cloning %s -> %s", src.repo, repo_dir)
            args = ["git", "clone"]
            if src.shallow:
                args += ["--depth", "1"]
                if src.ref:
                    args += ["--branch", src.ref]
            elif src.ref:
                args += ["--branch", src.ref]
            args += [src.repo, str(repo_dir)]
            subprocess.run(args, check=True)
        else:
            logger.info("Updating %s", repo_dir)
            # Fetch updates
            fetch_args = ["git", "-C", str(repo_dir), "fetch", "--all", "--tags"]
            if src.shallow:
                fetch_args += ["--depth", "1"]
            subprocess.run(fetch_args, check=True)

            # Checkout ref if provided, else fast-forward default
            if src.ref:
                subprocess.run(["git", "-C", str(repo_dir), "checkout", src.ref], check=True)
                subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
            else:
                subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)

        # Determine current commit
        rev = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"]).decode().strip()

        state = load_state()
        last = get_git_meta(state, src.repo)
        if last == rev:
            logger.info("Repo unchanged at %s", rev)
        else:
            logger.info("Repo advanced: %s -> %s", last, rev)
            set_git_meta(state, src.repo, rev)
            save_state(state)

        return repo_dir
