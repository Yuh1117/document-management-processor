import tempfile

TEMP_DIR = tempfile.gettempdir()
LAPLACIAN_VAR_THRESHOLD = 70.0

MIN_IMAGE_WIDTH = 300
MIN_IMAGE_HEIGHT = 300
MIN_CONTRAST_THRESHOLD = 15.0
VALIDATE_ALL_PDF_PAGES = False

TEXT_MIN_QUALITY_SCORE = 60
TEXT_MAX_INVALID_CHAR_RATIO = 0.3

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOC_MIME = "application/msword"
TXT_MIME = "text/plain"
PDF_MIME = "application/pdf"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
XLS_MIME = "application/vnd.ms-excel"

MIME_TO_EXT: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    PDF_MIME: "pdf",
    DOCX_MIME: "docx",
    DOC_MIME: "doc",
    TXT_MIME: "txt",
    XLSX_MIME: "xlsx",
    XLS_MIME: "xls",
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")

MIME_TO_CONTENT_TYPE: dict[str, str] = {
    TXT_MIME: "txt",
    DOCX_MIME: "docx",
    DOC_MIME: "doc",
    PDF_MIME: "pdf",
    XLSX_MIME: "xlsx",
    XLS_MIME: "xls",
}

EXT_TO_CONTENT_TYPE: dict[str, str] = {
    ".txt": "txt",
    ".docx": "docx",
    ".doc": "doc",
    ".pdf": "pdf",
    ".xlsx": "xlsx",
    ".xls": "xls",
    **{ext: "image" for ext in IMAGE_EXTENSIONS},
}
TEXT_FILE_ENCODINGS = ("utf-8", "utf-8-sig", "cp1258", "latin-1")
PDF_MIN_CHARS_PER_PAGE = 50
