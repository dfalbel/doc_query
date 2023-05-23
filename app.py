from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Generator, Sequence

import chromadb  # pyright: ignore[reportMissingTypeStubs]
import chromadb.api  # pyright: ignore[reportMissingTypeStubs]
import pypdf
import shiny.experimental as x
import tiktoken
from chromadb.api import Collection  # pyright: ignore[reportMissingTypeStubs]
from chromadb.config import Settings  # pyright: ignore[reportMissingTypeStubs]
from langchain.text_splitter import RecursiveCharacterTextSplitter
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

import chatstream
from chatstream import openai_types

# TODO: Make it so this runs only on connect (not on a local computer)
torch_cache_dir = Path.home() / ".cache" / "torch"
if not torch_cache_dir.exists():
    if not torch_cache_dir.parent.exists():
        torch_cache_dir.parent.mkdir(parents=True)

    local_torch_cache_dir = Path(__file__).parent / "torch_cache"
    if not local_torch_cache_dir.exists():
        local_torch_cache_dir.mkdir()

    print(f"Creating symlink {torch_cache_dir} to {local_torch_cache_dir}")
    torch_cache_dir.symlink_to(local_torch_cache_dir)


# Maximum number of context chunks to send to the API.
N_DOCUMENTS = 30
# Maximum number of tokens in the context chunks to send to the API.
CONTEXT_TOKEN_LIMIT = 3200
# Print debugging info to the console
DEBUG = True

# Avoid the following warning:
# huggingface/tokenizers: The current process just got forked, after parallelism has
# already been used. Disabling parallelism to avoid deadlocks...
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Code for initializing popper.js tooltips.
tooltip_init_js = """
var tooltipTriggerList = [].slice.call(
  document.querySelectorAll('[data-bs-toggle="tooltip"]')
);
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
  return new bootstrap.Tooltip(tooltipTriggerEl);
});
"""

app_ui = x.ui.page_fillable(
    ui.head_content(ui.tags.title("Posit Document Query")),
    x.ui.layout_sidebar(
        x.ui.sidebar(
            ui.h4("Posit Document Query"),
            ui.hr(),
            ui.output_ui("collection_selector_ui"),
            ui.hr(),
            ui.input_select("model", "Model", choices=["gpt-3.5-turbo"]),
            ui.p(ui.h5("Export conversation")),
            ui.input_radio_buttons(
                "download_format", None, ["Markdown", "JSON"], inline=True
            ),
            ui.div(
                ui.download_button("download_conversation", "Download"),
            ),
            ui.hr(),
            ui.p(
                "Built with ",
                ui.a("Shiny for Python", href="https://shiny.rstudio.com/py/"),
            ),
            ui.p(
                ui.a(
                    "Source code",
                    href="https://github.com/wch/chatstream",
                    target="_blank",
                ),
            ),
            width=280,
            position="right",
        ),
        chatstream.chat_ui("chat1"),
    ),
    # Initialize the tooltips at the bottom of the page (after the content is in the DOM)
    ui.tags.script(tooltip_init_js),
)

# ======================================================================================
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str((Path(__file__).parent / "doc_db").resolve()),
    )
)


def server(input: Inputs, output: Outputs, session: Session):
    # collection: reactive.Value[Collection] = reactive.Value()

    @output
    @render.ui
    def collection_selector_ui():
        collections = chroma_client.list_collections()
        if len(collections) == 0:
            return ui.div(
                {"class": "mx-auto text-center"},
                ui.h4("No collections found. Upload a file to get started..."),
            )

        return ui.input_radio_buttons(
            "collection",
            ui.strong("Document collection"),
            choices=[collection.name for collection in collections],
        )

    @reactive.Calc
    def collection() -> Collection:
        return chroma_client.get_collection(  # pyright: ignore[reportUnknownMemberType]
            input.collection()
        )

    def add_context_to_query(query: str) -> str:
        results = collection().query(
            query_texts=[query],
            n_results=min(collection().count(), N_DOCUMENTS),
        )

        if results["documents"] is None:
            context = "No context found"
        else:
            token_count = 0
            context = ""
            for doc in results["documents"][0]:
                result_token_count = get_token_count(doc, input.model())
                if token_count + result_token_count >= CONTEXT_TOKEN_LIMIT:
                    break

                token_count += result_token_count
                context += doc + "\n\n"

        prompt_template = f"""Use these pieces of context to answer the question at the end.
        You can also integrate other information that you know.
        If you don't know the answer, say that you don't know; don't try to make up an answer.

        {context}

        Question: {query}

        Answer:
        """

        if DEBUG:
            print(json.dumps(results, indent=2))
            print(prompt_template)

        return prompt_template

    chat_session = chatstream.chat_server(
        "chat1",
        model=input.model,
        query_preprocessor=add_context_to_query,
        debug=True,
    )

    def download_conversation_filename() -> str:
        if input.download_format() == "JSON":
            ext = "json"
        else:
            ext = "md"
        return f"conversation-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.{ext}"

    @session.download(filename=download_conversation_filename)
    def download_conversation() -> Generator[str, None, None]:
        if input.download_format() == "JSON":
            res = chatstream.chat_messages_enriched_to_chat_messages(
                chat_session.session_messages()
            )
            yield json.dumps(res, indent=2)

        else:
            yield chat_messages_to_md(chat_session.session_messages())


app = App(app_ui, server)


# ======================================================================================
# Utility functions
# ======================================================================================


def chat_messages_to_md(messages: Sequence[openai_types.ChatMessage]) -> str:
    """
    Convert a list of ChatMessage objects to a Markdown string.

    Parameters
    ----------
    messages
        A list of ChatMessageobjects.

    Returns
    -------
    str
        A Markdown string representing the conversation.
    """
    res = ""

    for message in messages:
        if message["role"] == "system":
            # Don't show system messages.
            continue

        res += f"## {message['role'].capitalize()}\n\n"
        res += message["content"]
        res += "\n\n"

    return res


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    with open(pdf_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)

        lines: list[str] = []
        for i in range(len(pdf_reader.pages)):
            lines.append(pdf_reader.pages[i].extract_text())

    return "\n".join(lines)


async def add_file_content_to_db(
    collection: chromadb.api.Collection,
    file: str | Path,
    label: str,
    debug: bool = False,
) -> None:
    file = Path(file)

    with ui.Progress(min=1, max=15) as p:
        p.set(message="Extracting text...")
        await asyncio.sleep(0)
        if file.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file)
        else:
            text = file.read_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        text_chunks = text_splitter.split_text(text)

    if debug:
        print(json.dumps(text_chunks, indent=2))

    with ui.Progress(min=1, max=len(text_chunks)) as p:
        for i in range(len(text_chunks)):
            p.set(value=i, message="Adding text to database...")
            await asyncio.sleep(0)
            collection.add(
                documents=text_chunks[i],
                metadatas={"filename": label, "page": str(i)},
                ids=f"{label}-{i}",
            )


def get_token_count(s: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))
