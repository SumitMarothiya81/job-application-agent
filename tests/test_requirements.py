"""Guards for reproducible installs."""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent


def lines(name: str) -> list[str]:
    return [ln.strip() for ln in (ROOT / name).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_lock_file_pins_every_package_exactly():
    pinned = lines("requirements.lock")
    assert len(pinned) > 60 and all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][^\s]*", ln) for ln in pinned)
    names = [canon(ln.split("==")[0]) for ln in pinned]
    assert len(names) == len(set(names))                                   # no duplicates


def test_every_direct_dependency_is_bounded_and_present_in_the_lock():
    locked = {canon(ln.split("==")[0]) for ln in lines("requirements.lock")}
    for ln in lines("requirements.txt"):
        req = Requirement(ln)
        assert req.specifier and ">=" in str(req.specifier), ln                 # has a lower bound
        assert req.name.lower() == "tzdata" or "<" in str(req.specifier), ln     # and an upper bound (tzdata is data only)
        assert canon(req.name) in locked, f"{req.name} is missing from requirements.lock"


def test_the_installed_versions_satisfy_the_declared_ranges():
    for ln in lines("requirements.txt"):
        req = Requirement(ln)
        try:
            installed = version(req.name)
        except PackageNotFoundError:
            pytest.skip(f"{req.name} is not installed here")
        assert req.specifier.contains(installed, prereleases=True), f"{req.name} {installed} is outside {req.specifier}"


def test_the_dockerfile_installs_from_the_lock_file():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY requirements.lock ." in dockerfile and "pip install -r requirements.lock" in dockerfile
