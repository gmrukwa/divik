import importlib
import sys

try:
    import gin
    from absl import flags
    _HAS_GIN = True
except ImportError:
    _HAS_GIN = False


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
    import gin
    from absl import flags
    flags.DEFINE_multi_string(
        'config', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string(
        'param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    gin.parse_config_files_and_bindings(FLAGS.config, FLAGS.param)


def configurable(klass):
    """Marks class as configurable via gin-config"""
    if _HAS_GIN:
        klass.__init__ = gin.external_configurable(
            klass.__init__, name=klass.__name__, blacklist=['self'])
    return klass
