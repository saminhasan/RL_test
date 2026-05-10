import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def _build_flags() -> tuple[list[str], list[str]]:
    if os.name == "nt":
        # /Ox and /GL are aggressive MSVC optimizations; /LTCG enables whole-program optimization at link time.
        compile_args = ["/O2", "/Ox", "/Ob2", "/Oi", "/Ot", "/GL", "/fp:fast"]
        link_args = ["/LTCG"]
    else:
        # Aggressive native tuning for local builds.
        compile_args = ["-O3", "-march=native", "-mtune=native", "-ffast-math", "-funroll-loops", "-flto"]
        link_args = ["-flto"]
    return compile_args, link_args


compile_args, link_args = _build_flags()


ext_modules = [
    Extension(
        name="cy_rules",
        sources=["cy_rules.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]


setup(
    name="mnsp-rl-cy-rules",
    ext_modules=cythonize(
        ext_modules,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
            "infer_types": True,
        },
    ),
)
# python setup.py build_ext --inplace