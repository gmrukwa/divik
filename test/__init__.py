import logging


log_format = '%(asctime)s [%(levelname)s] %(filename)40s:%(lineno)3s'\
             + ' - %(funcName)40s\t%(message)s'
logging.basicConfig(format=log_format, level=logging.DEBUG)
