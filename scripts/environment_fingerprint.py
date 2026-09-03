"""Environment fingerprint — research vs deployment baseline (P0.9.5).

Read-only diagnostic. Prints Python / platform / architecture plus installed
versions of the packages used by the research pipeline (src/analyzer,
src/evaluation) and the Streamlit app (app/streamlit_app.py). Also reports
Prophet's bundled CmdStan version and the MD5 of its precompiled
prophet_model.bin, so that two environments can be compared at the binary
level (Prophet 1.3.0 vs 1.4.0 bundle the same CmdStan 2.37.0 but ship
differently compiled binaries).

Design rules:
- stdlib only (importlib.metadata); packages are queried by their recorded
  metadata and never imported, so this script stays fast and cannot crash on
  a partially installed environment.
- Missing packages are reported as NOT INSTALLED — never an exception.
- No usernames, no absolute home paths, no API keys / tokens / secrets.

Usage:
    python3 scripts/environment_fingerprint.py
"""

import hashlib
import importlib.metadata as md
import os
import platform
import sys

# Ordered by the project's dependency layers: runtime / research / deploy.
PACKAGES = [
    # core runtime
    "pandas",
    "numpy",
    "scipy",
    # research pipeline (src/analyzer + src/evaluation)
    "scikit-learn",
    "xgboost",
    "torch",
    "prophet",
    "cmdstanpy",
    "stanio",
    "holidays",
    # app / deployment (app/streamlit_app.py)
    "streamlit",
    "plotly",
    "openpyxl",
    "akshare",
    # tooling
    "setuptools",
    "pip",
]


def pkg_version(dist_name: str) -> str:
    try:
        return md.version(dist_name)
    except md.PackageNotFoundError:
        return "NOT INSTALLED"


def prophet_stan_metadata():
    """Locate Prophet's bundled CmdStan and precompiled binary via dist files.

    Returns (cmdstan_version, bin_relpath, bin_md5, stan_relpath, stan_md5).
    relpaths are relative to the prophet distribution root, so no absolute
    home / site-packages path leaks into output. Any unknown piece is
    reported as UNKNOWN rather than guessed.
    """
    try:
        dist = md.distribution("prophet")
    except md.PackageNotFoundError:
        return ("NOT INSTALLED", "", "", "", "")
    root = dist.locate_file("prophet")

    # CmdStan version: directory named cmdstan-<version> under prophet/stan_model
    cmdstan_version = "UNKNOWN"
    stan_model = root / "stan_model"
    if stan_model.is_dir():
        for child in sorted(os.listdir(stan_model)):
            if child.startswith("cmdstan-"):
                cmdstan_version = child[len("cmdstan-"):]
                break

    def rel_and_md5(rel):
        p = dist.locate_file(rel)
        if not p.is_file():
            return "", ""
        with open(p, "rb") as fh:
            return rel, hashlib.md5(fh.read()).hexdigest()

    bin_rel, bin_md5 = rel_and_md5("prophet/stan_model/prophet_model.bin")
    stan_rel, stan_md5 = rel_and_md5("prophet/stan_model/prophet.stan")
    return cmdstan_version, bin_rel, bin_md5, stan_rel, stan_md5


def main():
    lines = []
    lines.append(f"python_version = {platform.python_version()}")
    lines.append(f"python_impl    = {platform.python_implementation()}")
    lines.append(f"platform       = {platform.platform()}")
    lines.append(f"machine        = {platform.machine()}")
    lines.append(f"executable     = {os.path.basename(sys.executable) or 'UNKNOWN'}")
    lines.append("")
    lines.append("# packages (metadata version; NOT INSTALLED if absent)")
    for name in PACKAGES:
        lines.append(f"{name:<14} = {pkg_version(name)}")
    lines.append("")
    lines.append("# prophet bundled stan artifacts")
    cmdstan_v, bin_rel, bin_md5, stan_rel, stan_md5 = prophet_stan_metadata()
    lines.append(f"prophet_cmdstan_version = {cmdstan_v}")
    lines.append(f"prophet_model_bin       = {bin_rel}  (md5 {bin_md5 or 'UNKNOWN'})")
    lines.append(f"prophet_stan            = {stan_rel}  (md5 {stan_md5 or 'UNKNOWN'})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
