import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4


LOG_DIR = Path(__file__).resolve().parent / "logs"

CHINA_TIME = timezone(timedelta(hours=8))


def new_session_id():
    """每次启动程序，生成一个新的会话编号。"""
    return uuid4().hex


def write_event(session_id, event_type, data):
    """将一条运行事件追加到当前会话的日志文件。"""

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.now(CHINA_TIME).isoformat(),
        "session_id": session_id,
        "event_type": event_type,
        "data": data,
    }

    file_path = LOG_DIR / f"{session_id}.jsonl"

    with file_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(event, ensure_ascii=False) + "\n"
        )

    return file_path