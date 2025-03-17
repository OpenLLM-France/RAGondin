from collections import defaultdict
from pathlib import Path
from components.indexer.loaders.BaseLoader import BaseLoader
from langchain_core.documents.base import Document
from langchain_community.document_loaders import UnstructuredPowerPointLoader

class CustomPPTLoader(BaseLoader):
    doc_loaders = {
        '.pptx': UnstructuredPowerPointLoader, 
        '.ppt': UnstructuredPowerPointLoader
    }

    cat2md = {
        'Title': '#',
        'NarrativeText': '>',
        'ListItem': '-',
        'UncategorizedText': '',
        'EmailAddress': '',
        'FigureCaption': '*'
    }
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.loader_args = kwargs
        self.page_sep = page_sep
    
    def group_by_hierarchy(self, docs: list[Document]):
        """Group related elements within each page by parent_id and category_depth"""
        grouped_elements = defaultdict(list)
        
        # Sort by depth to ensure proper hierarchy
        sorted_docs = sorted(docs, key=lambda x: x.metadata.get('category_depth', 0))
        
        for doc in sorted_docs:
            parent_id = doc.metadata.get('parent_id', None)
            if parent_id:
                grouped_elements[parent_id].append(doc)
            else:
                grouped_elements[doc.metadata.get('element_id')].append(doc)
        return grouped_elements
    
    async def aload_document(self, file_path, metadata: dict = None):
        path = Path(file_path)
        cls_loader = CustomPPTLoader.doc_loaders.get(path.suffix, None)

        if cls_loader is None:
            raise Exception(f"This loader only supports {CustomPPTLoader.doc_loaders.keys()} format")
        
        loader = cls_loader(
            file_path=str(file_path), 
            mode='elements',
            **self.loader_args
        )
        elements = await loader.aload()

        # Step 1: Group elements by page_number
        grouped_by_page = defaultdict(list)
        for doc in elements:
            page_number = doc.metadata.get('page_number')
            grouped_by_page[page_number].append(doc)
        
        # Final structure: dictionary of pages, each containing grouped elements
        final_grouped_structure = {}
        for page, docs in grouped_by_page.items():
            final_grouped_structure[page] = self.group_by_hierarchy(docs)

        content = ''
        for page, grouped_elements in final_grouped_structure.items():
            t = ''
            for v in grouped_elements.values():
                for elem in v:
                    meta = elem.metadata
                    cat_md = CustomPPTLoader.cat2md.get(meta.get("category"), '')
                    t += f"{cat_md} {elem.page_content}\n"

            if page is not None and content:
                t = f'{self.page_sep}' + t
            
            content += t

        return Document(
            page_content=content, 
            metadata=metadata
        )