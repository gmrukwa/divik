"""gin-config compatibility"""
import sys

try:
    import gin
    from absl import flags
    import divik.core._gin_bugfix
    _HAS_GIN = True
except ImportError:
    _HAS_GIN = False


MISSING_GIN_ERROR = """
gin-config package missing. You should install divik with appropriate extras:
    pip install divik[gin]
"""


def parse_gin_args():
    """Parse arguments with gin-config

    If you have `gin` extras installed, you can call `parse_gin_args`
    to parse command line arguments or config files to configure
    your runs.

    Command line arguments are used like `--param='DiviK.k_max=50'`.
    Config files are passed via `--config=path.gin`.

    More about format of `.gin` files can be found here:
    https://github.com/google/gin-config
    """
    try:
        import gin
        from absl import flags
    except ImportError as ex:
        raise ImportError(MISSING_GIN_ERROR) from ex
    flags.DEFINE_multi_string(
        'config', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string(
        'param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    gin.parse_config_files_and_bindings(FLAGS.config, FLAGS.param)


def configurable(klass, *args, **kwargs):
    """Marks class as configurable via gin-config"""
    if _HAS_GIN:
        klass = gin.configurable(klass, *args, **kwargs)
    return klass
