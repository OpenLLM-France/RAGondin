from pathlib import Path
from langchain_core.documents.base import Document


def load_sys_template(file_path: Path) -> tuple[str, str]:
    with open(file_path, mode="r") as f:
        sys_msg = f.read()
        return sys_msg


def format_context(docs: list[Document]) -> str:
    """Build context string from list of documents."""
    sources = []
    context = "Extracted documents:\n"
    for i, doc in enumerate(docs, start=1):
        context += f"""
        Document: {doc.page_content}
        [doc_{i}]: {doc.metadata["source"]}#page={doc.metadata["page"]+1}
        =======\n
        """
        sources.append(
            (f"[doc_{i}]", doc.metadata["source"], doc.metadata["page"]+1)
        )
    return context, sources