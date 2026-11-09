"""Attachment-level Indico extraction for the parent-child pipeline.

Reuses legacy authentication and text extraction, but does not access ChromaDB.
"""

import os
from urllib.parse import urljoin, urlparse

from src.extractors.indico_extractor_multithreaded import IndicoExtractor


class IndicoParentChildExtractor(IndicoExtractor):
    """Extract Indico attachments without chunking or database access."""

    def __init__(self):
        super().__init__(chroma=None)

    def _scrape_event_attachments_html(self, event_id):
        """Discover supported attachments while retaining contribution metadata."""
        import re

        url = f"{self.base_url}/event/{event_id}/"
        soup = self.get_soup_parser(url)
        found = []
        seen = set()

        for item in soup.select("li.timetable-item"):
            title_element = item.select_one(".timetable-title")
            speaker_element = item.select_one(".speaker-list")

            title = (
                title_element.get_text(strip=True)
                if title_element else None
            )
            speaker = (
                speaker_element.get_text(" ", strip=True)
                if speaker_element else "Unknown Author"
            )
            speaker = re.sub(r"(?i)^speakers?\s*:\s*", "", speaker)
            speaker = re.sub(r"\s*\([^)]*\)", "", speaker).strip()

            for anchor in item.select('a[href*="/attachments/"]'):
                href = anchor.get("href") or ""
                if not re.search(
                    r"\.(pdf|pptx|docx|xls|xlsx)(?:\?|$)",
                    href, re.I
                ):
                    continue

                attachment_url = urljoin(self.base_url, href)
                if attachment_url in seen:
                    continue

                seen.add(attachment_url)
                filename = os.path.basename(urlparse(attachment_url).path)

                found.append({
                    "title": title or filename,
                    "author": speaker or "Unknown Author",
                    "url": attachment_url,
                    "filename": filename,
                    "file_type": filename.rsplit(".", 1)[-1].lower(),
                })

        # Also discover attachments outside timetable items.
        for anchor in soup.select('a[href*="/attachments/"]'):
            href = anchor.get("href") or ""
            if not re.search(
                r"\.(pdf|pptx|docx|xls|xlsx)(?:\?|$)",
                href, re.I
            ):
                continue

            attachment_url = urljoin(self.base_url, href)
            if attachment_url in seen:
                continue

            seen.add(attachment_url)
            filename = os.path.basename(urlparse(attachment_url).path)

            found.append({
                "title": filename,
                "author": "Unknown Author",
                "url": attachment_url,
                "filename": filename,
                "file_type": filename.rsplit(".", 1)[-1].lower(),
            })

        return found

    def _attachment_candidates(self, obj):
        """Collect attachments from common Indico export structures."""
        found = []

        if not isinstance(obj, dict):
            return found

        for attachment in obj.get("attachments") or []:
            if isinstance(attachment, dict):
                found.append(attachment)

        for folder in obj.get("folders") or []:
            if isinstance(folder, dict):
                for attachment in folder.get("attachments") or []:
                    if isinstance(attachment, dict):
                        found.append(attachment)

        for key in ("material", "materials"):
            materials = obj.get(key) or []
            if isinstance(materials, dict):
                materials = [materials]

            for material in materials:
                if isinstance(material, dict):
                    for attachment in material.get("attachments") or []:
                        if isinstance(attachment, dict):
                            found.append(attachment)

        return found

    def _normalize_files(self, attachment):
        """Return one candidate for each actual attached file."""
        files = attachment.get("files") or []
        if not isinstance(files, list):
            files = []

        entries = files if files else [attachment]
        normalized = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            url = (
                entry.get("download_url")
                or entry.get("url")
                or entry.get("link")
                or attachment.get("download_url")
                or attachment.get("url")
                or attachment.get("link")
            )
            if not isinstance(url, str) or not url:
                continue

            url = urljoin(self.base_url + "/", url)
            filename = (
                entry.get("filename")
                or attachment.get("filename")
                or os.path.basename(urlparse(url).path)
                or attachment.get("title")
                or "unknown"
            )

            normalized.append({
                "url": url,
                "filename": str(filename),
                "content_type": (
                    entry.get("content_type")
                    or attachment.get("content_type")
                    or ""
                ),
            })

        return normalized

    def _extract_file(self, candidate, metadata):
        """Download and extract one file, preserving its metadata."""
        url = candidate["url"]

        try:
            content, _ = self._download_file(
                url, self.session, self.max_file_bytes
            )
            if not content:
                return None

            content_type = candidate.get("content_type") or ""
            if not content_type:
                extension = os.path.splitext(
                    urlparse(url).path
                )[1].lower()
                content_type = {
                    ".pdf": "application/pdf",
                    ".pptx": (
                        "application/vnd.openxmlformats-officedocument."
                        "presentationml.presentation"
                    ),
                    ".docx": (
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    ".xls": "application/vnd.ms-excel",
                    ".xlsx": (
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                }.get(extension, "")

            if not content_type:
                return None

            raw_text, document_type = self.get_raw_text(
                content_type, content
            )

            if not isinstance(raw_text, str) or not raw_text.strip():
                return None

            return {
                **metadata,
                "attachment_url": url,
                "filename": candidate["filename"],
                "document_type": document_type,
                "cleaned_text": raw_text.strip(),
            }

        except Exception as exc:
            print(f"Indico attachment extraction failed: {url}: {exc}")
            return None

    def _fetch_category_events(self, categ_id: int, limit: int, start: int):
        """Fetch a bounded category export without modifying the legacy extractor."""
        if limit == 0:
            return []

        url = f"{self.base_url}/export/categ/{categ_id}.json"
        response = self.session.get(
            url,
            params={"limit": max(1, limit), "start": max(0, start)},
            timeout=(5, 15),
        )
        response.raise_for_status()

        if self._is_cloudflare_challenge(response):
            raise RuntimeError(f"Cloudflare blocked category {categ_id}")

        results = response.json().get("results") or []
        return results[:limit] if limit > 0 else results

    def iter_event_attachments(self, event):
        """Yield extracted attachments for one Indico event."""
        event_id = event.get("id")
        if event_id is None:
            return

        event_url = f"{self.base_url}/event/{event_id}/"
        start = event.get("startDate") or {}

        metadata = {
            "indico_id": str(event_id),
            "event_url": event_url,
            "meeting_name": event.get("title") or "Unknown Title",
            "conveners": self.get_authors(event),
            "start_date": start.get("date") or "",
            "start_time": start.get("time") or "",
            "location": str(event.get("location") or ""),
            "event_description": str(event.get("description") or ""),
            "source": "indico",
        }

        details = self._fetch_event_details(event_id)
        seen_urls = set()
        extracted_any = False

        def process(obj, extra_metadata):
            nonlocal extracted_any

            for attachment in self._attachment_candidates(obj):
                for candidate in self._normalize_files(attachment):
                    url = candidate["url"]
                    if url in seen_urls:
                        continue

                    record = self._extract_file(
                        candidate, {**metadata, **extra_metadata}
                    )
                    if record:
                        seen_urls.add(url)
                        extracted_any = True
                        yield record

        yield from process(details, {
            "presentation_title": metadata["meeting_name"],
            "speaker_name": "Unknown Author",
        })

        for contribution in details.get("contributions") or []:
            if not isinstance(contribution, dict):
                continue

            yield from process(contribution, {
                "presentation_title": (
                    contribution.get("title")
                    or metadata["meeting_name"]
                ),
                "speaker_name": self._extract_speakers(contribution),
            })

        if not extracted_any:
            for attachment in self._scrape_event_attachments_html(event_id):
                candidate = {
                    "url": attachment["url"],
                    "filename": attachment.get("filename") or "unknown",
                    "content_type": "",
                }
                if candidate["url"] in seen_urls:
                    continue

                record = self._extract_file(candidate, {
                    **metadata,
                    "presentation_title": (
                        attachment.get("title")
                        or metadata["meeting_name"]
                    ),
                    "speaker_name": (
                        attachment.get("author")
                        or "Unknown Author"
                    ),
                })
                if record:
                    seen_urls.add(candidate["url"])
                    yield record
