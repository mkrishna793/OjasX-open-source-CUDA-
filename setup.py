import os
import shutil
import warnings
from setuptools import setup

def has_cpp_compiler() -> bool:
    if os.environ.get("OJASX_NO_CPP", "0") == "1":
        return False
    # Check for standard C++ compilers
    for compiler in ["cl", "g++", "clang++", "icx"]:
        if shutil.which(compiler) is not None:
            return True
    return False

ext_modules = []
cmdclass = {}

if has_cpp_compiler():
    try:
        from torch.utils.cpp_extension import BuildExtension, CppExtension
        ext_modules = [
            CppExtension(
                name="torchcl._C",
                sources=["torchcl/csrc/torchcl_extension.cpp"],
            )
        ]
        cmdclass["build_ext"] = BuildExtension
    except Exception as e:
        warnings.warn(f"[OjasX] Could not initialize CppExtension: {e}. Building pure-Python package.")

setup(
    name="ojasx",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
