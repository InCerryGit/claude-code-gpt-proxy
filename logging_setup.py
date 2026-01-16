import glob
import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


class MessageFilter(logging.Filter):
    def filter(self, record):
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator",
        ]

        if hasattr(record, "msg") and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True


class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings."""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)


def setup_logging():
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.WARNING)
    if not isinstance(log_level, int):
        log_level = logging.WARNING

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    formatter = ColorizedFormatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file = os.environ.get("LOG_FILE")
    if log_file:
        log_file = log_file.replace("\\", "/")
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        backup_count_env = os.environ.get("LOG_FILE_BACKUP_COUNT", "7")
        log_file_pattern = os.environ.get("LOG_FILE_PATTERN")
        if log_file_pattern is not None:
            log_file_pattern = log_file_pattern.strip() or None
        try:
            backup_count = int(backup_count_env)
        except ValueError:
            backup_count = 7

        log_file_rendered = datetime.now().strftime(log_file)
        file_handler = TimedRotatingFileHandler(
            log_file_rendered,
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8",
            utc=False,
        )
        file_handler.suffix = "%Y%m%d"

        def rotate_namer(name):
            rollover_token = os.path.basename(name).split(".")[-1]
            if log_file_pattern:
                rollover_date = datetime.strptime(rollover_token, "%Y%m%d")
                rotated_basename = rollover_date.strftime(log_file_pattern)
                return os.path.join(os.path.dirname(log_file_rendered), rotated_basename)
            base_name = os.path.basename(log_file_rendered)
            return os.path.join(
                os.path.dirname(log_file_rendered),
                f"{base_name}-{rollover_token}",
            )

        file_handler.namer = rotate_namer

        stale_glob = os.path.join(os.path.dirname(log_file_rendered), "*%Y%m%d*")
        for stale_path in glob.glob(stale_glob):
            if os.path.basename(stale_path) != os.path.basename(log_file_rendered):
                try:
                    os.remove(stale_path)
                except OSError:
                    pass

        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)

    root_logger.addFilter(MessageFilter())


class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"


def log_request_beautifully(
    method, path, claude_model, openai_model, num_messages, num_tools, status_code
):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"

    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]

    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"

    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"

    status_str = (
        f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}"
        if status_code == 200
        else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    )

    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"

    print(log_line)
    print(model_line)
