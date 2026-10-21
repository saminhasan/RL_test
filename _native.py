"""Loads the C++ engine and solver (cpp/msw/) through cppyy.

Windows: cppyy 3.1.2 compiles with its own clang 13, which cannot parse the
C++ standard library headers of MSVC 14.30+ (VS 2022 / VS 2026). So before
cppyy is imported, INCLUDE is pointed at an older toolset (MSVC 14.29, the
"MSVC v142 - VS 2019 C++ x64/x86 build tools" component of VS Build Tools)
plus the Windows SDK. Set MSW_MSVC_INCLUDE to a ...\\MSVC\\<version>\\include
directory to choose the toolset yourself.

cppyy 3.5.0 does not build on Windows; pin cppyy==3.1.2.
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys

import numpy as np

_HERE    = os.path.dirname(os.path.abspath(__file__))
_CPP_DIR = os.path.join(_HERE, "cpp")


def _version(name: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in name.split("."))
    except ValueError:
        return ()


def _windows_include() -> str:
    roots = {
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("ProgramFiles", r"C:\Program Files"),
    }

    msvc = os.environ.get("MSW_MSVC_INCLUDE")
    if not msvc:
        found = []
        for root in roots:
            pattern = os.path.join(root, "Microsoft Visual Studio", "*", "*", "VC", "Tools", "MSVC", "*", "include")
            for inc in glob.glob(pattern):
                ver = _version(os.path.basename(os.path.dirname(inc)))
                if ver and ver < (14, 30):
                    found.append((ver, inc))
        if not found:
            raise ImportError(
                "cppyy needs the MSVC 14.29 (VS 2019) C++ headers; newer ones do not parse.\n"
                "Install 'MSVC v142 - VS 2019 C++ x64/x86 build tools' from the Visual Studio "
                "Installer, or set MSW_MSVC_INCLUDE to an older ...\\MSVC\\<version>\\include."
            )
        msvc = max(found)[1]

    sdks = []
    for root in roots:
        for inc in glob.glob(os.path.join(root, "Windows Kits", "10", "Include", "10.*")):
            if os.path.isdir(os.path.join(inc, "ucrt")):
                sdks.append((_version(os.path.basename(inc)), inc))
    if not sdks:
        raise ImportError("Windows 10/11 SDK headers not found (Windows Kits\\10\\Include).")
    sdk = max(sdks)[1]

    return ";".join([msvc] + [os.path.join(sdk, d) for d in ("ucrt", "um", "shared")])


def _load_system_runtime() -> None:
    """cppyy ships an old copy of the MSVC C++ runtime (msvcp140.dll & co). If that
    copy is the first one loaded, PyTorch's c10.dll fails to start later in the
    same process ("DLL initialization routine failed"). Loading the system's newer
    copies first makes both work."""
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll"):
        path = os.path.join(system32, name)
        if os.path.exists(path):
            try:
                ctypes.WinDLL(path)
            except OSError:
                pass


if sys.platform == "win32":
    os.environ["INCLUDE"] = _windows_include()
    _load_system_runtime()

import cppyy  # noqa: E402  (must come after INCLUDE is set)

cppyy.add_include_path(_CPP_DIR)
cppyy.include("msw/engine.hpp")

msw = cppyy.gbl.msw


def view(owner: object, addr: int, dtype: np.dtype | type, count: int) -> np.ndarray:
    """Zero-copy numpy view of `count` items at `addr`. The view keeps `owner` alive."""
    dtype = np.dtype(dtype)
    buf = (ctypes.c_ubyte * (count * dtype.itemsize)).from_address(addr)
    buf._owner = owner
    return np.frombuffer(buf, dtype=dtype, count=count)
