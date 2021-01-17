"""gin-config compatibility"""
import os
import sys

try:
    import gin

    import divik.core._gin_bugfix

    _HAS_GIN = True
except ImportError:
    _HAS_GIN = False


MISSING_GIN_ERROR = """
gin-config package missing. You should install divik with appropriate extras:
    pip install divik[gin]
"""


def dump_gin_args(destination):
    """Dump gin-config effective configuration

    If you have `gin` extras installed, you can call `dump_gin_args`
    save effective gin configuration to a file.
    """
    try:
        import gin
    except ImportError as ex:
        raise ImportError(MISSING_GIN_ERROR) from ex
    with open(os.path.join(destination, "config.gin"), "w") as outfile:
        outfile.write(gin.operative_config_str())


if _HAS_GIN:
    configurable = gin.configurable
else:

    def configurable(name_or_fn=None, *args, **kwargs):
        if name_or_fn is None:
            return lambda x: x
        return name_or_fn


def parse_args():
    """Parse gin config files and parameter overrides from command line"""
    import argparse

    import gin

    parser = argparse.ArgumentParser()

    parser.add_argument("--param", nargs="*", help="List of Gin parameter bindings")
    parser.add_argument("--config", nargs="*", help="List of paths to the config files")

    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.config, args.param)
