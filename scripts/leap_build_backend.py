"""scikit-build-core, with the Ninja generator for AMD GPU builds on Windows.

CMake's Visual Studio generators, scikit-build-core's default on Windows, do
not support HIP. When LEAP_GPU is AMD on Windows and no generator is chosen,
-GNinja is added to CMAKE_ARGS, where scikit-build-core also sees it when it
decides whether the ninja package is a build requirement. Every other build
goes through scikit-build-core unchanged.
"""

import os
import re
import shlex
import sys

from scikit_build_core import build as _build
from scikit_build_core.build import *  # noqa: F401,F403


def _use_ninja(config_settings, state):
    if sys.platform != "win32":
        return
    from scikit_build_core.builder.generator import parse_generator
    from scikit_build_core.settings.skbuild_read_settings import SettingsReader

    settings = SettingsReader.from_file("pyproject.toml", config_settings, state=state).settings
    args = [*settings.cmake.args, *shlex.split(os.environ.get("CMAKE_ARGS", ""))]
    gpu = str(settings.cmake.define.get("LEAP_GPU", ""))
    for prev, arg in zip(["", *args], args):
        if prev == "-D" or arg.startswith("-D"):
            match = re.fullmatch(r"(?:-D)?LEAP_GPU(?::\w+)?=(.*)", arg)
            gpu = match.group(1) if match else gpu
    chosen = parse_generator(args) or "CMAKE_GENERATOR" in settings.cmake.define or os.environ.get("CMAKE_GENERATOR")
    if re.fullmatch("amd", gpu, re.IGNORECASE) and not chosen:
        os.environ["CMAKE_ARGS"] = " ".join(filter(None, [os.environ.get("CMAKE_ARGS"), "-GNinja"]))


def get_requires_for_build_wheel(config_settings=None):
    _use_ninja(config_settings, "wheel")
    return _build.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings=None):
    _use_ninja(config_settings, "editable")
    return _build.get_requires_for_build_editable(config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _use_ninja(config_settings, "wheel")
    return _build.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _use_ninja(config_settings, "editable")
    return _build.build_editable(wheel_directory, config_settings, metadata_directory)
