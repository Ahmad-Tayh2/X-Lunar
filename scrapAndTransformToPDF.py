import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Create directories for saving files if they don't exist
os.makedirs("downloads/html", exist_ok=True)
os.makedirs("downloads/docs", exist_ok=True)

visited = set()

def save_html(url, content):
    # Generate a filename from the URL, ensure it ends with .html
    file_name = url.split("/")[-1] or "index.html"
    if not file_name.endswith(".html"):
        file_name += ".html"
    
    file_path = os.path.join("downloads/html", file_name)
    
    # Make sure to use only valid filename characters
    file_path = file_path.replace(":", "_").replace("?", "_").replace("&", "_")
    
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

def save_file(url, file_type):
    # Generate a filename from the URL and ensure it has the correct extension
    file_name = url.split("/")[-1]
    if not file_name.endswith(file_type):
        file_name += file_type
    
    # Create directory for docs if necessary and save the file
    if file_type in ['.pdf', '.doc', '.docx']:
        file_path = os.path.join("downloads/docs", file_name)
        
        # Download the file
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Saved: {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

def crawl(url, depth=1):
    if depth == 0 or url in visited:
        return
    visited.add(url)

    print(f"Crawling: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Save the HTML content to file
        save_html(url, response.text)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Get and print the page title
        print("Title:", soup.title.string if soup.title else "No Title")

        # Look for links that are PDFs, DOC, or DOCX and save them
        for link_tag in soup.find_all('a', href=True):
            full_url = urljoin(url, link_tag['href'])
            
            if full_url.startswith("http://utm.rnu.tn"):
                # Check if it's a PDF or DOC file
                if full_url.lower().endswith('.pdf'):
                    save_file(full_url, '.pdf')
                elif full_url.lower().endswith(('.doc', '.docx')):
                    save_file(full_url, '.docx')

                # Crawl the link for more content
                crawl(full_url, depth=depth - 1)

    except requests.exceptions.RequestException as e:
        print(f"Failed to crawl {url}: {e}")

# Start crawling
start_url = "https://fst.rnu.tn/fr/bibliotheque"
crawl(start_url, depth=2)  # depth=2 to include sublinks one level deep
