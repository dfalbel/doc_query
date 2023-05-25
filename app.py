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

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

import chatstream
from chatstream import openai_types


model_id = "stabilityai/stablelm-tuned-alpha-3b"
#model_id = "EleutherAi/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Maximum number of context chunks to send to the API.
N_DOCUMENTS = 30
# Maximum number of tokens in the context chunks to send to the API.
CONTEXT_TOKEN_LIMIT = 200
# Print debugging info to the console
DEBUG = True

app_ui = x.ui.page_fillable(
    ui.head_content(ui.tags.title("Posit Document Query")),
    x.ui.layout_sidebar(
        x.ui.sidebar(
            ui.h4("Posit Document Query"),
            ui.hr(),
            ui.output_ui("collection_selector_ui"),
            ui.hr(),
            ui.input_select("model", "Model", choices=["gpt-3.5-turbo"]),
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
        ui.div(
            ui.output_ui("messages_ui"),
            x.ui.layout_column_wrap(
                0.5,
                ui.input_text("query", "", width="100%"),
                ui.input_action_button("send", "Send", width="100%"),
            )
        )
    ),
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
    
    messages = reactive.Value([])
    generating = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.send)
    def _():
        updated = messages() + [{'role': 'user', 'content': input.query()}]
        messages.set(updated)
        generating.set(True)

    @output(priority=10)
    @render.ui
    def messages_ui():
        print("rendering messages")
        return ui_messages(messages())
    
    streamer = reactive.Value()
    count_tokens = reactive.Value(0)

    @reactive.Effect
    @reactive.event(generating)
    def _():
        if not generating():
            return
        query = add_context_to_query(input.query())
        inputs = tokenizer(query, return_tensors="pt")
        
        stream = TextIteratorStreamer(tokenizer, skip_prompt=True)
        generation_kwargs = dict(**inputs, streamer=stream, max_new_tokens=30)
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        streamer.set(stream)
        count_tokens.set(count_tokens()+1)
        messages.set(messages() + [{'role': 'bot', 'content': ''}])

    @reactive.Effect
    @reactive.event(count_tokens)
    def _():
        print("getting next token")
        try:
            token = next(streamer())
        except StopIteration:
            generating.set(False)
            return
        msgs = messages().copy()
        msgs[-1]['content'] += token
        messages.set(msgs)
        count_tokens.set(count_tokens()+1)

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


app = App(app_ui, server)


# ======================================================================================
# Utility functions
# ======================================================================================

def ui_messages (messages):
    return ui.div(*[
        x.ui.card(
            {'style': "margin-bottom:5px;"},
            x.ui.card_body(
                ui.p(message['content'])
            ),
        )
        for message in messages]
    )


def get_token_count(s: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))
