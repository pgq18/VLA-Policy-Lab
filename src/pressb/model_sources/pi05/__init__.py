"""Load the bundled policy under LeRobot's canonical π0.5 package name."""
from __future__ import annotations

from importlib.abc import MetaPathFinder
from importlib.util import spec_from_file_location
from pathlib import Path
import sys
from threading import RLock


_PACKAGE = "lerobot.policies.pi05"
_GEMMA = "lerobot.policies.pi_gemma"
_LOCK = RLock()


class _Pi05Finder(MetaPathFinder):
    def __init__(self, package_dir):
        self.package_dir = package_dir

    def find_spec(self, fullname, path=None, target=None):
        if fullname == _PACKAGE:
            return spec_from_file_location(fullname, self.package_dir / "__init__.py")
        if fullname == _GEMMA:
            return spec_from_file_location(fullname, self.package_dir.parent / "pi_gemma.py")
        return None


_finder = None


def activate_pi05_source(root=None):
    """Select the policy source without importing LeRobot or its ML dependencies.

    LeRobot's policy registry eagerly imports π0.5. If it has already run,
    retain registered classes only when their source matches the selected tree.
    """
    global _finder
    root = Path(__file__).resolve().parent if root is None else Path(root).expanduser().resolve()
    package_dir = root / "lerobot/policies/pi05"
    with _LOCK:
        if _finder is not None:
            if _finder.package_dir != package_dir:
                raise RuntimeError(
                    f"π0.5 source already selected at {_finder.package_dir}. "
                    f"Start a fresh process to select {package_dir}"
                )
            return root
        for filename in ("pi05/__init__.py", "pi05/configuration_pi05.py", "pi05/modeling_pi05.py",
                         "pi05/processor_pi05.py", "pi_gemma.py"):
            source = package_dir.parent / filename
            if not source.is_file():
                raise ValueError(f"Incomplete pi05 source tree: missing {source}")

        for name, module in sys.modules.copy().items():
            if module is None or (name not in (_PACKAGE, _GEMMA) and not name.startswith(_PACKAGE + ".")):
                continue
            relative_name = name.removeprefix("lerobot.policies.")
            expected = package_dir.parent.joinpath(*relative_name.split("."))
            if hasattr(module, "__path__"):
                expected = expected / "__init__.py"
            else:
                expected = expected.with_suffix(".py")
            origin = getattr(module, "__file__", None)
            if origin:
                actual = Path(origin).resolve()
                if actual == expected or (actual.is_file() and expected.is_file()
                                          and actual.read_bytes() == expected.read_bytes()):
                    continue
            raise RuntimeError(
                f"{name} is already imported from another source tree ({origin}); "
                "activate the selected π0.5 source in a fresh process before importing LeRobot policies"
            )

        _finder = _Pi05Finder(package_dir)
        sys.meta_path.insert(0, _finder)
        package = sys.modules.get(_PACKAGE)
        if package is not None:
            # Keep existing class and registry identities while routing future
            # child imports through the selected source directory.
            package.__path__ = [str(package_dir)]
            if package.__spec__ is not None:
                package.__spec__.submodule_search_locations = package.__path__
    return root
