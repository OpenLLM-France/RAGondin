from pydantic import BaseModel
from typing import Optional, List, Dict, Union

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5  # default to 5 if not provided