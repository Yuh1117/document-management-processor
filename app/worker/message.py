import json
from dataclasses import dataclass
from typing import Any


@dataclass
class DocumentMessage:
    doc_id: str
    file_url: str
    file_type: str = "application/octet-stream"
    owner_id: str | None = None
    name: str | None = None
    file_size: int | None = None

    @classmethod
    def from_body(cls, body: bytes | bytearray | memoryview) -> "DocumentMessage":
        raw = bytes(body) if isinstance(body, memoryview) else body
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        msg: dict[str, Any] = json.loads(text)
        return cls(
            doc_id=msg["doc_id"],
            file_url=msg["file_url"],
            file_type=msg.get("file_type", "application/octet-stream"),
            owner_id=msg.get("owner_id"),
            name=msg.get("name"),
            file_size=_to_int(msg.get("file_size")),
        )


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None
