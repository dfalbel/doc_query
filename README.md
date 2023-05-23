Documentation query app
=======================

This is a Shiny for Python application which allows you to ask questions about a set of documents. It uses the [chatstream](https://github.com/wch/chatstream/) package to query the OpenAI API.

First, install some Python packages:

```bash
pip install -r requirements.txt
```

Then load at least one set of text/markdown/PDF files into a vector database, using `doc_db.py` script. This will save the documents in a database in a directory named `doc_db/`.

The syntax is:

```
./doc_db.py add-dir <collection name> <directory> <file pattern>
```

For example:

```bash
./doc_db.py add-dir admin-guide ../docs/admin "*.qmd"
./doc_db.py add-dir user-guide ../docs/user "*.qmd"
```

After the documents are loaded into the vector database, they can be queried using the Shiny app.

To use the Shiny app, you need an API key from OpenAI. After you have it, you can set an environment variable named `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY="<your_openai_api_key>"
```

Then you can run the app with:

```bash
shiny run app.py --launch-browser
```
