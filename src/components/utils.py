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
    if not docs:
        return 'No document found from the database', []
    
    sources = []
    context = "Extracted documents:\n"

    for i, doc in enumerate(docs, start=1):
        doc_id = f"[doc_{i}]"

        document = f"""
        document id: {doc_id}
        content: \n{doc.page_content.strip()}\n
        """

        # document = f"""<chunk document_id={doc_id}>\n{doc.page_content.strip()}\n</chunk>\n"""
        # Source: {source} (Page: {page})
    
        context += document
        context += "=" * 40 + "\n\n"

        sources.append(
            {
                "doc_id": doc_id,
                'source': doc.metadata["source"],
                'sub_url_path': doc.metadata["sub_url_path"],
                'page': doc.metadata["page"],
                'content': doc.page_content
            }
        )

    return context, sources