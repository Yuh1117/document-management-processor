import os
import tempfile

WORKER_THREADS = 4

DEFAULT_LANG = "vi"

###

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONSTANTS_DIR = os.path.join(BASE_DIR, "constants")

ES_MAPPINGS_DIR = os.path.join(CONSTANTS_DIR, "es_mappings")
DOCUMENTS_INDEX_MAPPING_FILE = os.path.join(ES_MAPPINGS_DIR, "documents_index.json")

ES_QUERIES_DIR = os.path.join(CONSTANTS_DIR, "es_queries")
FULL_TEXT_QUERY_FILE = os.path.join(ES_QUERIES_DIR, "full_text.json")
SEMANTIC_QUERY_FILE = os.path.join(ES_QUERIES_DIR, "semantic.json")
HYBRID_QUERY_FILE = os.path.join(ES_QUERIES_DIR, "hybrid.json")

TEMP_DIR = tempfile.gettempdir()
LAPLACIAN_VAR_THRESHOLD = 70.0
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

MIN_IMAGE_WIDTH = 300
MIN_IMAGE_HEIGHT = 300
MIN_CONTRAST_THRESHOLD = 15.0
VALIDATE_ALL_PDF_PAGES = False

TEXT_MIN_QUALITY_SCORE = 60
TEXT_MAX_INVALID_CHAR_RATIO = 0.3

SEARCH_SEMANTIC_MIN_SCORE = 0.7
SEARCH_KNN_POOL_SIZE = 1000
SEARCH_RRF_K = 60
SEARCH_DEFAULT_CANDIDATE_SIZE = 100
SEARCH_CANDIDATE_MULTIPLIER = 10
SEARCH_MAX_CANDIDATE_SIZE = 1000

SNIPPET_MAX_CHARS = 320

###

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOC_MIME = "application/msword"
TXT_MIME = "text/plain"
PDF_MIME = "application/pdf"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
XLS_MIME = "application/vnd.ms-excel"

SUPPORTED_TYPES = (
    "image/*",
    PDF_MIME,
    TXT_MIME,
    DOC_MIME,
    DOCX_MIME,
    XLSX_MIME,
    XLS_MIME,
)

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
TEXT_FILE_ENCODINGS = ("utf-8", "utf-8-sig", "cp1258", "latin-1")
PDF_MIN_CHARS_PER_PAGE = 50
