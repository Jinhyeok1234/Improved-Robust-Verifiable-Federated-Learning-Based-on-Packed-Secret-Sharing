# setup.py
from setuptools import setup, Extension
import pybind11
import os
from pathlib import Path

if os.name == 'nt':  # Windows
    VCPKG_BASE = Path(r"C:\dev\vcpkg\installed\x64-windows")
    GMP_INCLUDE_DIR = str(VCPKG_BASE / "include")
    GMP_LIB_DIR = str(VCPKG_BASE / "lib")
    
    compile_args = ["/std:c++17", "/openmp", "/utf-8"] 
    link_args = []
    libraries = ["gmp", "gmpxx"]
else:  # Linux/macOS
    GMP_INCLUDE_DIR = "/usr/include"
    GMP_LIB_DIR = "/usr/lib/x86_64-linux-gnu"
    compile_args = ["-std=c++17", "-O3", "-fopenmp"]
    link_args = ["-fopenmp"]
    libraries = ["gmp", "gmpxx"]

# The elliptic-curve commitment backend is compiled only when the OpenSSL headers are
# present. Set VPSS_WITH_EC=0 to force it off, or =1 to force it on (e.g. when the headers
# live somewhere this check does not look).
def _openssl_present():
    forced = os.environ.get("VPSS_WITH_EC")
    if forced == "0":
        return False
    if forced == "1":
        return True
    candidates = [GMP_INCLUDE_DIR, "/usr/include", "/usr/local/include",
                  "/opt/homebrew/include", "/opt/local/include"]
    return any(os.path.exists(os.path.join(d, "openssl", "ec.h")) for d in candidates)

WITH_EC = _openssl_present()
if WITH_EC:
    compile_args.append("/DVPSS_WITH_EC" if os.name == "nt" else "-DVPSS_WITH_EC")
    libraries += ["libcrypto"] if os.name == "nt" else ["crypto"]
    print("[setup] elliptic-curve backend: ENABLED")
else:
    print("[setup] elliptic-curve backend: disabled (OpenSSL headers not found); "
          "the Z_p backend still builds")

ext_modules = [
    Extension(
        "vpss_module",
        sources=["bindings.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            GMP_INCLUDE_DIR
        ],
        library_dirs=[GMP_LIB_DIR],
        libraries=libraries,
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

setup(
    name="vpss_module",
    version="1.0",
    ext_modules=ext_modules,
)