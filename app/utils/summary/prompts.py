EXCERPT_SEPARATOR = "\n[...]\n"

SUMMARY_PROMPTS = {
    "vi": (
        "Bạn là trợ lý AI chuyên tóm tắt tài liệu. Hãy tóm tắt nội dung sau "
        "một cách ngắn gọn, rõ ràng và đầy đủ các ý chính.\n\n"
        "Yêu cầu:\n"
        "- Tóm tắt bằng tiếng Việt\n"
        "- Giữ lại các thông tin quan trọng, số liệu, tên riêng\n"
        "- Trình bày thành 1-3 đoạn văn mạch lạc, khoảng 150-300 từ\n"
        "- Chỉ dùng thông tin có trong nội dung, không suy diễn thêm\n\n"
        "Nội dung cần tóm tắt:\n---\n{text}\n---\n\nTóm tắt:"
    ),
    "en": (
        "You are an AI assistant specialized in document summarization. "
        "Summarize the following content concisely, clearly, and covering all key points.\n\n"
        "Requirements:\n"
        "- Summarize in English\n"
        "- Retain important information, figures, and proper nouns\n"
        "- Write 1-3 coherent paragraphs, about 150-300 words\n"
        "- Only use information present in the content, do not speculate\n\n"
        "Content to summarize:\n---\n{text}\n---\n\nSummary:"
    ),
}

# Used when the content is a set of excerpts chosen from a longer document.
EXCERPTS_NOTE = {
    "vi": (
        "Lưu ý: tài liệu dài nên nội dung dưới đây là các đoạn trích tiêu biểu "
        "theo đúng thứ tự trong tài liệu, phân cách bởi [...].\n\n"
    ),
    "en": (
        "Note: the document is long, so the content below consists of "
        "representative excerpts in document order, separated by [...].\n\n"
    ),
}

# Map step: summarize one part of a long document.
PART_PROMPTS = {
    "vi": (
        "Đây là phần {index}/{total} của một tài liệu dài. Hãy liệt kê ngắn gọn "
        "các ý chính, số liệu và tên riêng quan trọng của phần này bằng tiếng Việt, "
        "dạng gạch đầu dòng.\n\n---\n{text}\n---\n\nÝ chính:"
    ),
    "en": (
        "This is part {index}/{total} of a long document. Briefly list the key "
        "points, figures and important proper nouns of this part in English, "
        "as bullet points.\n\n---\n{text}\n---\n\nKey points:"
    ),
}

# Reduce step: merge the part summaries into the final summary.
COMBINE_NOTE = {
    "vi": (
        "Lưu ý: nội dung dưới đây là ý chính của từng phần trong một tài liệu dài, "
        "theo đúng thứ tự. Hãy viết một bản tóm tắt thống nhất cho toàn bộ tài liệu.\n\n"
    ),
    "en": (
        "Note: the content below is the key points of each part of a long document, "
        "in order. Write one unified summary of the whole document.\n\n"
    ),
}


def pick(prompts: dict[str, str], language: str, default: str) -> str:
    return prompts.get(language, prompts[default])
