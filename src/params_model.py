from pydantic import BaseModel
from typing import List

class ImageData(BaseModel):
    image: str
    id: int
    label: str = None

class QueryParams(BaseModel):
    collection: str
    query_filter: str
    output_fields: List[str]
    limit: int = 3