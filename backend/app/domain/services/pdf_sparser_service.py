import fitz  # PyMuPDF

def extract_pdf_metadata(file_content: bytes):
    doc = fitz.open(stream=file_content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    metadata = doc.metadata
    num_pages = len(doc)

    return {
        "text": text,
        "metadata": metadata,
        "num_pages": num_pages
    }
