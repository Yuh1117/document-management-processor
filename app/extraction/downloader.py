from __future__ import annotations

import logging
import os
import re
import tempfile

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import (
    R2_ACCESS_KEY,
    R2_ENDPOINT,
    R2_REGION,
    R2_SECRET_KEY,
)
from app.extraction.constants import (
    TEMP_DIR,
)
from app.extraction.file_types import safe_unlink, suffix_for_type

logger = logging.getLogger(__name__)


S3_URI = re.compile(r"^s3://([^/]+)/(.+)$")


class FileDownloader:
    """Downloads a remote file (S3 or HTTP) to a local temp path."""

    def __init__(self, temp_dir: str = TEMP_DIR) -> None:
        self.temp_dir = temp_dir

    def download(self, file_url: str, file_type: str) -> str:
        ext = suffix_for_type(file_type)
        fd, path = tempfile.mkstemp(suffix=f".{ext}", dir=self.temp_dir)
        os.close(fd)
        try:
            if file_url.startswith("s3://"):
                self.from_s3(file_url, path)
            else:
                self.from_http(file_url, path)
        except Exception:
            safe_unlink(path)
            raise
        return path

    def from_s3(self, file_url: str, path: str) -> None:
        m = S3_URI.match(file_url)
        if not m:
            raise ValueError(f"Invalid S3 URI: {file_url!r}")
        bucket, key = m.group(1), m.group(2)
        try:
            boto3.client(
                "s3",
                aws_access_key_id=R2_ACCESS_KEY or None,
                aws_secret_access_key=R2_SECRET_KEY or None,
                region_name=R2_REGION,
                endpoint_url=R2_ENDPOINT or None,
            ).download_file(bucket, key, path)
        except (ClientError, BotoCoreError):
            logger.exception("S3 download failed bucket=%s key=%s", bucket, key)
            raise

    @staticmethod
    def from_http(file_url: str, path: str) -> None:
        resp = requests.get(file_url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
