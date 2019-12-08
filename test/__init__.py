import logging


try:
    import divik._matlab_legacy

    logger = logging.getLogger(divik._matlab_legacy.__name__)
    logger.setLevel(logging.ERROR)
except ImportError:
    pass  # In environments without MATLAB this should work as well


log_format = '%(asctime)s [%(levelname)s] %(filename)40s:%(lineno)3s'\
             + ' - %(funcName)40s\t%(message)s'
logging.basicConfig(format=log_format, level=logging.DEBUG)
