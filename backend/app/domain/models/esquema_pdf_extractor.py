from pydantic import BaseModel
from typing import Dict


class PDFMetadata(BaseModel):
    sha256: str
    num_pages: int
    extracted_text: str
    metadata: Dict
