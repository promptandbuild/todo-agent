import logging
import os
from colorama import Fore, Back, Style, init as colorama_init
from pyfiglet import figlet_format

# Initialize colorama (for Windows compatibility)
colorama_init(autoreset=True)

# Map environment log levels to logging constants
LOG_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

# Custom formatter with color and emoji
class FancyFormatter(logging.Formatter):
    def format(self, record):
        level = record.levelname
        msg = record.getMessage()

        if level == "INFO":
            return f"{Fore.GREEN}{Style.BRIGHT}[INFO] {msg}{Style.RESET_ALL} "
        elif level == "WARNING":
            return f"{Fore.YELLOW}{Style.BRIGHT}[⚠ WARNING]{Style.RESET_ALL} {Style.DIM}{msg}{Style.RESET_ALL}"
        elif level == "ERROR":
            return f"{Fore.RED}{Style.BRIGHT}[✖ ERROR]{Style.RESET_ALL} {Style.BRIGHT}{msg}{Style.RESET_ALL}"
        elif level == "DEBUG":
            return f"{Fore.CYAN}{Style.DIM}[DEBUG]{Style.RESET_ALL} {msg}"
        elif level == "CRITICAL":
            return f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}[CRITICAL]{Style.RESET_ALL} {msg.upper()}"
        else:
            return msg

# Function 1: Get a logger instance
def get_fancy_logger(name="FancyLogger"):
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = LOG_LEVEL_MAP.get(env_level, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False  # Avoid duplicated logs

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(FancyFormatter())
        logger.addHandler(ch)

    return logger

# Function 2: Print a styled banner
def print_banner(text: str, font: str = "slant", color: str = Fore.MAGENTA):
    banner = figlet_format(text, font=font)
    print(color + Style.BRIGHT + banner)


if __name__ == "__main__":
    # Example usage
    logger = get_fancy_logger("MyAppLogger")
    print_banner("Welcome to MyApp", font="starwars", color=Fore.BLUE)

    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.debug("This is a debug message.")
    logger.critical("This is a critical message.")
    print_banner("Goodbye!", font="starwars", color=Fore.RED)
