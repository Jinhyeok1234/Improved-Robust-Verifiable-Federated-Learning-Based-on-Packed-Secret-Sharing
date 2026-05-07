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