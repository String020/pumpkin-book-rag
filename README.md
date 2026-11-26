# Pumpkin Book RAG

A Streamlit chatbot for asking machine-learning questions in Chinese about Datawhale's *Pumpkin Book*. It streams answers, keeps conversation history for the current session, and prompts the model to cite the PDF page numbers of the retrieved passages or say when the material is insufficient.

## How it works

PyMuPDF extracts the text and LangChain splits it into chunks of up to 500 characters with a 50-character overlap setting. OpenAI `text-embedding-3-small` embeddings are stored in a local Chroma index. For each question, the app retrieves the three closest chunks and passes them to the chat model. Follow-up questions are first rewritten using the conversation history. The default chat model is `gpt-6-luna`, configurable through `OPENAI_CHAT_MODEL`.

## Run locally

Use Python 3.12 and an OpenAI API key with access to the models.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest https://github.com/datawhalechina/pumpkin-book/releases/download/v2.0.0/pumpkin_book.pdf -OutFile data/pumpkin_book.pdf
```

Set `OPENAI_API_KEY` in `.env`, then run:

```powershell
.\.venv\Scripts\python.exe document_loader.py --preview
.\.venv\Scripts\python.exe document_loader.py
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The preview runs offline. The first index build sends 20 chunks to OpenAI. An existing complete local index is reused without embedding the PDF again. Chat sends questions and retrieved text to OpenAI, so chat requests can incur API charges.

## Data and limitations

The sample indexes **the first 20 chunks**, from PDF pages 1–4 of v2.0.0. These pages mainly contain introductory material, so the demo has limited coverage of the book's machine-learning topics. PDF extraction can lose mathematical notation.

Datawhale publishes the [Pumpkin Book v2.0.0 PDF](https://github.com/datawhalechina/pumpkin-book/releases/tag/v2.0.0) under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The PDF and the generated Chroma index stay local.
