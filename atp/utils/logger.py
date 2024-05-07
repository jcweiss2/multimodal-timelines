import logging
import os


class MakeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        dirname = os.path.dirname(filename)
        if dirname:  # dirname can be empty string
            os.makedirs(dirname, exist_ok=True)
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)


def init_logger(log_dir, phase='train', file_append=False):
    """ Initialize the root logger """
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) == 0, \
           "This method should be called before initializing the root logger"
    root_logger.setLevel(logging.DEBUG)

    filename = os.path.join(log_dir, f'{phase}_log.txt')
    file_handler = logging.FileHandler(filename,
                                       mode="a" if file_append else "w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    )
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    return root_logger


