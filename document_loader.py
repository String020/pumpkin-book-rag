import argparse
import hashlib
import re

import pymupdf
from chromadb.errors import NotFoundError
from filelock import FileLock, Timeout
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrieval import (
    CHUNK_LIMIT,
    EMBEDDING_MODEL,
    INDEX_DIR,
    PROJECT_DIR,
    SetupError,
    check_openai_key,
    open_index,
)

PDF_PATH = PROJECT_DIR / "data" / "pumpkin_book.pdf"


def load_chunks():
    if not PDF_PATH.is_file():
        raise SetupError("Save the v2.0.0 PDF as data/pumpkin_book.pdf (see README).")

    pages = []
    with pymupdf.open(PDF_PATH) as pdf:
        for page in pdf:
            text = re.sub(
                r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])",
                "",
                page.get_text(),
            ).strip()
            pages.append(Document(page_content=text, metadata={"page": page.number + 1}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)[:CHUNK_LIMIT]
    if len(chunks) != CHUNK_LIMIT:
        raise SetupError(f"Expected at least {CHUNK_LIMIT} chunks in the PDF.")
    return chunks


def index_pdf():
    with FileLock(str(INDEX_DIR.with_suffix(".lock")), timeout=0):
        try:
            existing_ids = set(open_index().get(include=[])["ids"])
        except NotFoundError:
            existing_ids = set()

        if len(existing_ids) == CHUNK_LIMIT:
            print(f"Reusing the local index ({CHUNK_LIMIT} chunks). No API calls made.")
            return

        chunks = load_chunks()
        with PDF_PATH.open("rb") as pdf:
            source_hash = hashlib.file_digest(pdf, "sha256").hexdigest()
        chunks_by_id = {f"{source_hash}-{i}": chunk for i, chunk in enumerate(chunks)}
        if not existing_ids.issubset(chunks_by_id):
            raise SetupError("The local index contains unexpected IDs. Move it aside before rebuilding.")

        pending_ids = [chunk_id for chunk_id in chunks_by_id if chunk_id not in existing_ids]
        check_openai_key()
        store = open_index(OpenAIEmbeddings(model=EMBEDDING_MODEL), create=True)
        store.add_documents([chunks_by_id[chunk_id] for chunk_id in pending_ids], ids=pending_ids)
        print(f"Indexed {len(pending_ids)} new chunks. Local total: {CHUNK_LIMIT}.")


def main():
    parser = argparse.ArgumentParser(description="Build the Pumpkin Book demo index.")
    parser.add_argument("--preview", action="store_true", help="Read and split the PDF offline.")
    args = parser.parse_args()

    try:
        if args.preview:
            chunks = load_chunks()
            pages = ", ".join(str(page) for page in sorted({chunk.metadata["page"] for chunk in chunks}))
            print(f"{len(chunks)} chunks ready from PDF pages {pages}. No API calls made.")
        else:
            index_pdf()
    except SetupError as error:
        parser.exit(1, f"{error}\n")
    except Timeout:
        parser.exit(1, "Another indexing process is running. Try again after it finishes.\n")


if __name__ == "__main__":
    main()
