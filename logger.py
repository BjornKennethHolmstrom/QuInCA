import logging
import sys

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[0;36m',  # Cyan
        'INFO': '\033[0;32m',   # Green
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',  # Red
        'CRITICAL': '\033[0;35m',  # Magenta
        'RESET': '\033[0m',  # Reset
    }

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}{log_message}{self.COLORS['RESET']}"

def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

# Usage
# logger = setup_logger(__name__, logging.DEBUG)
