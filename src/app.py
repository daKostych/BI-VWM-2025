# Standard library imports
import time
import re

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
from bs4 import BeautifulSoup
from document_db import DocumentsDB
#from src.config import BASE_DIR

from pathlib import Path

BASE_DIR = Path().resolve()

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(
    directory="/home/senyaaa/Work/bi-vwm.21/src/templates"
)

path = BASE_DIR / 'datasets' / 'pickled_data_2000.pkl'
# Initialize document database
db = DocumentsDB(
    data_path=path,
    load_pickled_data=True
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Return the main page with a list of all documents."""
    # Get all documents from the database
    documents = db.documents_dict

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "documents": documents}
    )


@app.get("/document/{id_}", response_class=HTMLResponse)
async def document(request: Request, id_: int) -> HTMLResponse:
    """Return the page for a specific document including similar documents and performance metrics."""
    # Get Wikipedia ID from database
    #document_internal_id = int(document_to_parse.split(",")[0][1:])
    wikipedia_id = db.documents_dict[id_][0]

    # Extract page ID from Wikipedia URL
    page_id = wikipedia_id.split("wikipedia-")[-1]

    # Construct Wikipedia URL
    url = f"https://en.wikipedia.org/?curid={page_id}"

    # Fetch Wikipedia content
    start_time = time.time()
    content = await fetch_wikipedia_content(url)

    # Measure time to fetch Wikipedia content
    fetch_time = round(time.time() - start_time, 2) if "Failed to retrieve Wikipedia page" not in content else 0

    similar_documents = db.get_similar_documents(id_)
    sequential_search_speed = db.get_speed_statistics().search_speed

    db.use_inverted_index = True
    similar_documents2 = db.get_similar_documents(id_)
    inverted_search_speed = db.get_speed_statistics().search_speed
    db.use_inverted_index = False

    assert similar_documents == similar_documents2, "Different recommendations"

    def extract_title(text: str) -> str:
        """Extract title from document text by finding words until two consecutive spaces."""
        match = re.search(r"^(.*?)(  |$)", text)  # Match until two consecutive spaces or end of text
        return match.group(1).strip() if match else "Untitled"

    similar_documents = [
        (doc_id, extract_title(db.documents_dict[doc_id][1])) for doc_id in similar_documents
    ]

    return templates.TemplateResponse(
        "document.html",
        {
            "request": request,
            "page_id": page_id,
            "content": content,
            "similar_documents": similar_documents,
            "similar_documents_speed": round(sequential_search_speed, 2),
            "similar_documents_speed_inverted": round(inverted_search_speed, 2),
            "time_to_fetch_wikipedia": fetch_time,
            "url": url
        }
    )


async def fetch_wikipedia_content(url: str) -> str:
    """Fetch and process Wikipedia page content."""
    try:
        start_time = time.time()
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()

        # Extract main content
        article = soup.find('div', class_='mw-parser-output')

        if article:
            return str(article)
        else:
            return "Failed to retrieve Wikipedia page content."

    except requests.exceptions.RequestException as e:
        return f"Failed to retrieve Wikipedia page. Error: {str(e)}"
    finally:
        end_time = time.time()
        print(f"Time to fetch Wikipedia content: {round(end_time - start_time, 2)} seconds")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
