#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    import chromadb  # pyright: ignore[reportMissingTypeStubs]
    import chromadb.api  # pyright: ignore[reportMissingTypeStubs]


# Avoid the following warning:
# huggingface/tokenizers: The current process just got forked, after parallelism has
# already been used. Disabling parallelism to avoid deadlocks...
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

OUTPUT_DIR = (Path(__file__).parent / "doc_db").resolve()


@click.group()
def main():
    pass


@main.command()
@click.argument("collection_name")
@click.argument("files", nargs=-1)
def add(collection_name: str, files: str):
    """Add files to the collection."""
    client = get_chromadb_client()
    collection = (
        client.get_or_create_collection(  # pyright: ignore[reportUnknownMemberType]
            collection_name
        )
    )
    for file in files:
        add_file_content_to_db(collection, file, Path(file).name)


@main.command(
    help="""Find files (recursively) in a directory and add them to the collection.

COLLECTION_NAME is the name of the collection to add the files to. This is an identifier used in the database for this set of files.

DIR is the directory to search in, recursively.

PATTERN is a glob pattern, e.g. "*.md". Note that it may need to be in quotes to avoid shell expansion.

\b
Example:
    vector_db.py add-dir user_docs docs/user/ "*.md"
"""
)
@click.argument("collection_name")
@click.argument("dir")
@click.argument("pattern")
def add_dir(collection_name: str, dir: str, pattern: str):
    dir_p = Path(dir).resolve()
    files: list[Path] = find_files(dir_p, pattern)

    client = get_chromadb_client()
    collection = (
        client.get_or_create_collection(  # pyright: ignore[reportUnknownMemberType]
            collection_name
        )
    )
    for file in files:
        rel_path = file.relative_to(dir_p)
        print(rel_path)
        add_file_content_to_db(collection, file, str(rel_path))


@main.command()
@click.argument("collection_name")
def delete(collection_name: str):
    """Delete a collection."""
    client = get_chromadb_client()
    client.delete_collection(collection_name)


@main.command("list")
@click.argument("collection_name", required=False)
def list_(collection_name: str | None = None):
    """List all collections."""
    client = get_chromadb_client()
    if collection_name is None:
        collections = client.list_collections()
        click.echo("Collections:\n  " + "\n  ".join([x.name for x in collections]))


# ======================================================================================


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    import pypdf

    with open(pdf_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)

        lines: list[str] = []
        for i in range(len(pdf_reader.pages)):
            lines.append(pdf_reader.pages[i].extract_text())

    return "\n".join(lines)


def add_file_content_to_db(
    collection: chromadb.api.Collection,
    file: str | Path,
    label: str,
    debug: bool = False,
) -> None:
    from langchain.text_splitter import (
        MarkdownTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    file = Path(file)

    if file.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(file)
    else:
        text = file.read_text()

    if file.suffix.lower() in (".md", ".rmd", ".qmd"):
        text_splitter = MarkdownTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    text_chunks = text_splitter.split_text(text)

    if debug:
        print(json.dumps(text_chunks, indent=2))

    for i in range(len(text_chunks)):
        collection.add(
            documents=text_chunks[i],
            metadatas={"filename": label, "page": str(i)},
            ids=f"{label}-{i}",
        )


def find_files(directory: Path | str, pattern: str) -> list[Path]:
    """
    Finds all files in a directory (and its subdirectories) matching a filename pattern.

    Returned paths are absolute.
    """
    directory = Path(directory)
    files: list[Path] = []

    for file_path in directory.rglob(pattern):
        files.append(file_path.resolve())

    return files


def get_chromadb_client():
    import chromadb  # pyright: ignore[reportMissingTypeStubs]
    import chromadb.config  # pyright: ignore[reportMissingTypeStubs]

    return chromadb.Client(
        chromadb.config.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(OUTPUT_DIR),
        )
    )


# ======================================================================================

if __name__ == "__main__":
    main()
