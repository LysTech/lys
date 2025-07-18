import logging
import json
import io
import pytest
from utils.json_logging import JsonFormatter, setup_json_logging

@pytest.fixture
def log_stream():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test_logger")
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield logger, stream
    logger.handlers = []


def test_json_formatter_basic_message(log_stream):
    logger, stream = log_stream
    logger.info("hello world")
    stream.seek(0)
    log_json = json.loads(stream.getvalue())
    assert log_json["level"] == "INFO"
    assert log_json["message"] == "hello world"
    assert log_json["name"] == "test_logger"
    assert "time" in log_json


def test_json_formatter_with_extra(log_stream):
    logger, stream = log_stream
    logger.info("something happened", extra={"user": "alice", "action": "start"})
    stream.seek(0)
    log_json = json.loads(stream.getvalue())
    assert log_json["user"] == "alice"
    assert log_json["action"] == "start"
    assert log_json["message"] == "something happened"


def test_json_formatter_multiple_logs(log_stream):
    logger, stream = log_stream
    logger.info("first", extra={"step": 1})
    logger.info("second", extra={"step": 2})
    stream.seek(0)
    logs = stream.getvalue().splitlines()
    assert len(logs) == 2
    first = json.loads(logs[0])
    second = json.loads(logs[1])
    assert first["step"] == 1
    assert second["step"] == 2
    assert first["message"] == "first"
    assert second["message"] == "second" 