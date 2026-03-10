# Research Paper Aggregator

A simple Streamlit application that searches across multiple research databases and displays results with external links to full papers.

## Features

- **Multi-source search**: arXiv, Semantic Scholar, OpenAlex
- **Dual search modes**: By topic/title or by author name
- **Clean interface**: View abstracts and metadata
- **External links**: Click to view full papers at source

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

3. Open your browser to the URL shown in terminal (usually http://localhost:8501)

## Usage

1. Choose search type: "Topic/Title" or "Author"
2. Enter your search query
3. Select max results (5-30)
4. Click "Search"
5. Browse results and click "View Full Paper" for external links

## Supported APIs

- **arXiv**: Open access papers in physics, math, CS, etc.
- **Semantic Scholar**: AI-powered academic search
- **OpenAlex**: Open catalog of scholarly papers

No API keys required - uses public endpoints.