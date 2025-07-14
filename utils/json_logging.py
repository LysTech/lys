import logging
import json

class JsonFormatter(logging.Formatter):
    """Format log records as JSON objects."""
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "time": self.formatTime(record, self.datefmt),
            "name": record.name,
            "message": record.getMessage(),
        }
        # Add any extra fields passed via 'extra'
        for key, value in record.__dict__.items():
            if key not in ("levelname", "msg", "args", "name", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "message", "asctime"):
                log_record[key] = value
        return json.dumps(log_record)

def setup_json_logging(level=logging.INFO):
    """Configure root logger to use JSON formatting."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler]) 