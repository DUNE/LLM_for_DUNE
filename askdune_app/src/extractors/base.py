from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import re
from scipy import ndimage
from src.utils.logger import get_logger
from io import BytesIO
import requests
import pymupdf
logger = get_logger(__name__)

PDF_TEXT_FLAGS = (
    pymupdf.TEXTFLAGS_TEXT
    & ~pymupdf.TEXT_PRESERVE_LIGATURES
    | pymupdf.TEXT_DEHYPHENATE
)

class BaseExtractor(ABC):
    """Base class for document extractors"""

    def __init__(self):
        super().__init__()

        pymupdf.set_messages(pylogging=True, pylogging_logger=logger)

    @abstractmethod
    def extract_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Extract documents from the source"""
        pass

    def get_chunks(self, text, chunk_size):
        if not text: return []
        start = 0
        end=chunk_size
        chunks = []
        while start < len(text):
            chunks.append(' '.join(text[start:end]))
            start=end

            end += chunk_size
        return chunks

    def _download_file(self, link, session, max_file_bytes) -> Optional[Tuple[bytes, Dict[str, str]]]:
        """Download a file with size check; returns (content, headers) or None if skipped/failure."""
        if not link:
            logger.error(f"Invalid link: {link}")
            return None, None
        try:
            assert not isinstance(session, str), type(session)
            # First try HEAD to get size
            try:
                head = session.head(link, allow_redirects=True)
                if head.status_code in (401, 403):
                    logger.warning(f"Unauthorized to download (HEAD): {link}")
                    raise requests.RequestException(f"HEAD returned {head.status_code}")
                if head.status_code == 404:
                    logger.warning(f"File not found (404): {link}")
                    return None, None
                if head.status_code >= 400:
                    raise requests.RequestException(f"HEAD returned {head.status_code}")

                size_hdr = head.headers.get("content-length", None)
                size = int(size_hdr) if (size_hdr and size_hdr.isdigit()) else 0
            except requests.RequestException:
                logger.debug(f"HEAD failed, falling back to GET for {link}")
                size = 0  # Unknown size, proceed to GET

            # GET request with browser User-Agent
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            resp = session.get(link, headers=headers, stream=True)
            if resp.status_code in (401, 403):
                logger.warning(f"Unauthorized to download (GET): {link}")
                return None, None
            if resp.status_code == 404:
                logger.warning(f"File not found (404): {link}")
                return None, None
            if resp.status_code >= 400:
                logger.error(f"GET failed ({resp.status_code}) for {link}")
                return None, None

            # Use content-length if HEAD succeeded, else unknown
            if size and size > max_file_bytes:
                logger.info(f"Skipping large file ({size} bytes > {max_file_bytes}): {link}")
                return None, None

            # Read content (cap if size unknown)
            if size and size <= max_file_bytes:
                logger.debug("Reading file")
                content = resp.content
            else:
                chunks = []
                downloaded_bytes = 0

                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        chunks.append(chunk)
                        downloaded_bytes += len(chunk)

                        if downloaded_bytes > max_file_bytes:
                            logger.info(f"Skipping file exceeding max size while streaming: {link}")
                            return None, None

                content = b"".join(chunks)

            return content, resp.headers

        except requests.RequestException as e:
            logger.error(f"Error downloading {link}: {e}")
            return None, None

    def get_raw_text(self, content_type, content):
        raw_text = ''
        document_type=''
        if 'application/pdf' in content_type:
            logger.debug("Getting from pdf")
            raw_text, document_type = self.extract_text_from_pdf(content)

            return raw_text, document_type

        elif ('presentationml.presentation' in content_type
                or 'application/vnd.ms-powerpoint' in content_type):
            raw_text,document_type = self.extract_text_from_pptx(content)
        elif ('wordprocessingml.document' in content_type
                or 'application/msword' in content_type
                or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type):
            raw_text,document_type = self.extract_text_from_docx(content)
        elif content_type.startswith('text/'):
            raw_text,document_type = self.extract_text_from_txt(content)
        elif ('spreadsheetml.sheet' in content_type
                or 'application/vnd.ms-excel' in content_type
                or 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type):
            raw_text,document_type = self.extract_text_from_xlsx(content)
        elif content_type in ('image/jpeg', 'image/png', 'image/jpg'):
            raw_text,document_type = self.extract_text_from_image(content)
        else:
            logger.warning(f"Unsupported content type ({content_type})")
        return raw_text, document_type

    def extract_text_from_pdf(self, pdf_content: bytes) -> tuple[str, str]:
        """Extract text from PDF content"""
        if not pdf_content:
            return "", ""

        try:
            with pymupdf.open(stream=pdf_content, filetype="pdf") as doc:
                if doc.page_count == 0:
                    return "", ""

                rect = doc[0].rect
                document_type = "slides" if rect.width >= rect.height else "document"

                try:
                    text = doc.to_markdown()
                except Exception:
                    logger.warning(
                        "Markdown extraction failed; falling back to plain text.",
                    )

                    text = "\n\n".join(
                        page.get_text(flags=PDF_TEXT_FLAGS, sort=True) for page in doc
                    )

                return text, document_type

        except Exception:
            logger.exception("Error extracting text from PDF")
            return "", ""

    def extract_text_from_pptx(self, pptx_content: bytes) -> str:
        """Extract text from PPTX content"""
        from pptx import Presentation

        text = ''
        try:
            presentation = Presentation(BytesIO(pptx_content))
            for slide in presentation.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        text += shape.text.replace('\n•', '\n- ') + '\n'
        except Exception as e:
            logger.error(f"Error extracting text from PPTX: {e}")
        return text, 'slides'

    def extract_text_from_docx(self, docx_content: bytes) -> str:
        """Extract text from DOCX/DOC content"""
        from docx import Document

        text = ''
        try:
            doc = Document(BytesIO(docx_content))
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + '\n'
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
        return text, 'document'

    def extract_text_from_txt(self, content: bytes) -> str:
        try:
            return content.decode("utf-8", errors="ignore"), 'document'
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {e}")
        return "", ''

    def extract_text_from_xlsx(self, content: bytes) -> tuple[str, str]:
        """Extract legacy XLS and modern XLSX spreadsheets as searchable text."""

        try:
            if not content:
                raise ValueError("Empty spreadsheet")

            lines = []

            # Legacy Excel 97-2003 binary format
            if content.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
                import xlrd

                workbook = xlrd.open_workbook(file_contents=content)

                for sheet in workbook.sheets():
                    lines.append(f"Sheet: {sheet.name}")

                    for row_index in range(sheet.nrows):
                        values = [
                            str(sheet.cell_value(row_index, col_index))
                            for col_index in range(sheet.ncols)
                            if sheet.cell_type(row_index, col_index)
                            not in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK)
                        ]

                        if values:
                            lines.append("\t".join(values))

            # Modern XLSX ZIP format
            elif content.startswith(b"PK"):
                from openpyxl import load_workbook

                workbook = load_workbook(
                    filename=BytesIO(content),
                    data_only=True,
                    read_only=True,
                )

                try:
                    for sheet in workbook.worksheets:
                        lines.append(f"Sheet: {sheet.title}")

                        for row in sheet.iter_rows(values_only=True):
                            values = [
                                str(cell)
                                for cell in row
                                if cell is not None
                            ]

                            if values:
                                lines.append("\t".join(values))

                finally:
                    workbook.close()

            else:
                raise ValueError("Unrecognized Excel file format")

            text = "\n".join(lines).strip()

            if not text:
                logger.warning("Spreadsheet contains no extractable text")
                return "", ""

            return text, "spreadsheet"

        except Exception as exc:
            logger.warning("Skipping unreadable spreadsheet: %s", exc)
            return "", ""

    def extract_text_from_image(self, image_content: bytes) -> tuple[str, str]:
        """Extract text from JPEG/PNG images using RapidOCR, unless disabled."""
        import os
        import time

        disable_ocr = os.getenv("DISABLE_OCR", "false").strip().lower() in (
            "true", "1", "yes", "on"
        )

        if disable_ocr:
            logger.info("OCR disabled by DISABLE_OCR=true")
            return "", "image"

        if not image_content:
            return "", "image"

        from rapidocr_onnxruntime import RapidOCR

        if not hasattr(self, "_image_ocr_engine"):
            init_start = time.perf_counter()

            try:
                ocr_threads = int(os.getenv("ASKDUNE_OCR_THREADS", "0"))
            except ValueError:
                logger.warning(
                    "Invalid ASKDUNE_OCR_THREADS; using RapidOCR defaults"
                )
                ocr_threads = 0

            if ocr_threads > 0:
                self._image_ocr_engine = RapidOCR(
                    intra_op_num_threads=ocr_threads,
                    inter_op_num_threads=1,
                )
            else:
                self._image_ocr_engine = RapidOCR()

            logger.info(
                "RapidOCR initialization time: %.2f seconds",
                time.perf_counter() - init_start,
            )

        start = time.perf_counter()
        result, _ = self._image_ocr_engine(image_content)
        elapsed = time.perf_counter() - start

        logger.info("RapidOCR processing time: %.2f seconds", elapsed)

        if not result:
            return "", "image"

        text = "\n".join(
            str(item[1]).strip()
            for item in result
            if len(item) > 1 and str(item[1]).strip()
        )

        logger.info("RapidOCR extracted %d characters", len(text))

        return text, "image"
