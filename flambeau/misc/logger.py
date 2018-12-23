import sys


class OutputLogger(object):
    """Output logger"""

    def __init__(self):
        self.file = None
        self.buffer = ''

    def set_log_file(self, filename, mode='wt'):
        assert self.file is None
        self.file = open(filename, mode)
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, data):
        if self.file is not None:
            self.file.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self):
        if self.file is not None:
            self.file.flush()


class TeeOutputStream(object):
    """Redirect output stream"""

    def __init__(self, child_streams, autoflush=False):
        self.child_streams = child_streams
        self.autoflush = autoflush

    def write(self, data):
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        for stream in self.child_streams:
            stream.write(data)
        if self.autoflush:
            self.flush()

    def flush(self):
        for stream in self.child_streams:
            stream.flush()


output_logger = None


def init_output_logging():
    """
    Initialize output logger
    """
    global output_logger
    if output_logger is None:
        output_logger = OutputLogger()
        sys.stdout = TeeOutputStream([sys.stdout, output_logger],
                                     autoflush=True)
        sys.stderr = TeeOutputStream([sys.stderr, output_logger],
                                     autoflush=True)


def set_output_log_file(filename, mode='wt'):
    """
    Set file name of output log

    :param filename: file name of log
    :type filename: str
    :param mode: the mode in which the file is opened
    :type mode: str
    """
    if output_logger is not None:
        output_logger.set_log_file(filename, mode)
