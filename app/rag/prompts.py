PROMPTS = {
    "vi": (
        "Bạn là trợ lý hỏi đáp về tài liệu của người dùng. Chỉ trả lời dựa trên các "
        "đoạn trích dưới đây, không dùng kiến thức bên ngoài. Nếu các đoạn trích không "
        "có thông tin để trả lời, hãy nói rõ là bạn không tìm thấy thông tin trong tài "
        "liệu. Trả lời ngắn gọn, rõ ràng.\n\n"
        "Các đoạn trích:\n{context}\n\n"
        "Câu hỏi: {question}\n\n"
        "Chỉ trả lời bằng tiếng Việt, kể cả khi câu hỏi hoặc đoạn trích dùng ngôn ngữ khác.\n"
        "Trả lời (bằng tiếng Việt):"
    ),
    "en": (
        "You are an assistant that answers questions about the user's documents. "
        "Answer only from the excerpts below and do not use outside knowledge. If the "
        "excerpts do not contain the answer, say clearly that you could not find it in "
        "the documents. Answer concisely.\n\n"
        "Excerpts:\n{context}\n\n"
        "Question: {question}\n\n"
        "Respond in English only, even if the question or excerpts are in another language.\n"
        "Answer (in English):"
    ),
}

SOURCE_LABEL = {"vi": "Tài liệu", "en": "Document"}

NO_ANSWER = {
    "vi": "Tôi không tìm thấy thông tin liên quan trong tài liệu của bạn.",
    "en": "I could not find relevant information in your documents.",
}

EMPTY_QUESTION = {
    "vi": "Vui lòng nhập câu hỏi.",
    "en": "Please enter a question.",
}
