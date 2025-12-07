"""
Custom FileLogger class for logging messages to a file. To separate console and file logging.
"""
import logging

class FileLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.logger = logging.getLogger("manual_logger")
        self.logger.setLevel(logging.INFO)
        # Create a file handler but don't attach it to root logging
        self.file_handler = logging.FileHandler(self.filename, "a")
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    def log(self, msg: str):
        # Manually emit the log
        self.file_handler.emit(logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg=msg,
            args=None,
            exc_info=None
        ))
