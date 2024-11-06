from pathlib import Path
from langchain_core.documents.base import Document

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # print("1st creation")
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        # else:
        #     print("Same one")
        return cls._instances[cls]
    

def load_sys_template(file_path: Path) -> tuple[str, str]:
    with open(file_path, mode="r") as f:
        sys_msg = f.read()
        return sys_msg
    

def format_context(docs: list[Document]) -> str:
    """Build context string from list of documents."""
    sources = []
    context = "Extracted documents:\n"

    for i, doc in enumerate(docs, start=1):
        doc_id = f"[doc_{i}]"
        source = doc.metadata["source"]
        page = doc.metadata["page"]

        document = f"""
        document id: {doc_id}
        content: \n{doc.page_content.strip()}\n
        """
        # Source: {source} (Page: {page})
    
        context += document
        context += "=" * 40 + "\n\n"

        sources.append(
            {
                "doc_id": doc_id,
                'source': source,
                'page': page,
                'content': doc.page_content
            }
        )

    return context, sources