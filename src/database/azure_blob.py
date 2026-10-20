"""
Azure Blob Storage integration for WatchGuard Vision.

Uploads security evidence images to the existing Azure Blob container.
Local evidence files remain untouched.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv


class AzureBlobSync:
    """Upload WatchGuard evidence images to Azure Blob Storage."""

    def __init__(self):
        load_dotenv()

        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv(
            "AZURE_STORAGE_CONTAINER",
            "watchguard-events",
        )

        self.enabled = bool(self.connection_string)

        self._service_client = None
        self._container_client = None

        if self.enabled:
            self._service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            self._container_client = self._service_client.get_container_client(
                self.container_name
            )

    def ensure_container(self) -> None:
        """Create the container if it does not already exist."""
        if not self.enabled or self._container_client is None:
            return

        try:
            self._container_client.create_container()
        except Exception as exc:
            # Container already exists is expected.
            if "ContainerAlreadyExists" not in str(exc):
                raise

    def upload_evidence(
        self,
        local_path: Optional[str],
        event_id: Optional[int] = None,
        risk_level: str = "GREEN",
        event_type: str = "EVENT",
    ) -> Optional[str]:
        """
        Upload one local evidence image to Azure Blob Storage.

        Returns the blob URL when uploaded, otherwise None.
        """
        if not self.enabled:
            return None

        if not local_path:
            return None

        source = Path(local_path)

        if not source.exists() or not source.is_file():
            return None

        self.ensure_container()

        now = datetime.now()
        safe_risk = str(risk_level).upper().replace(" ", "_")
        safe_event = str(event_type).upper().replace(" ", "_")

        blob_name = (
            f"{now:%Y/%m/%d}/"
            f"{safe_risk}/"
            f"{safe_event}/"
            f"{now:%H%M%S}_{event_id or uuid4().hex}.jpg"
        )

        content_type = "image/jpeg"
        suffix = source.suffix.lower()

        if suffix == ".png":
            content_type = "image/png"
            blob_name = blob_name.rsplit(".", 1)[0] + ".png"
        elif suffix == ".bmp":
            content_type = "image/bmp"
            blob_name = blob_name.rsplit(".", 1)[0] + ".bmp"

        blob_client = self._container_client.get_blob_client(blob_name)

        with source.open("rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=False,
                content_settings=ContentSettings(
                    content_type=content_type
                ),
            )

        return blob_client.url
