import os
import requests
import argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Create directory for saving HTML files if it doesn't exist
os.makedirs("downloads/html", exist_ok=True)
os.makedirs("downloads/docs", exist_ok=True)

visited = set()

def save_html(url, content, is_pdf_or_doc=False):
    # Generate a filename from the URL, ensure it ends with .html or .pdf/.doc
    file_name = url.split("/")[-1] or "index.html"
    if not file_name.endswith(".html") and not is_pdf_or_doc:
        file_name += ".html"
    elif is_pdf_or_doc:
        # Adjust filename for pdf or doc
        if not file_name.endswith(('.pdf', '.doc', '.docx')):
            file_name += ".pdf"
    
    # Choose directory based on file type
    file_dir = "downloads/html" if not is_pdf_or_doc else "downloads/docs"
    file_path = os.path.join(file_dir, file_name)
    
    # Make sure to use only valid filename characters
    file_path = file_path.replace(":", "_").replace("?", "_").replace("&", "_")
    
    # Write content to file
    if is_pdf_or_doc:
        with open(file_path, "wb") as file:  # Writing as binary for PDF/DOC files
            file.write(content)
    else:
        with open(file_path, "w", encoding="utf-8") as file:  # Writing as text for HTML files
            file.write(content)

def crawl(url):
    if url in visited:
        return
    visited.add(url)

    print(f"Crawling: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36'
        }
        # Disable SSL certificate verification by setting verify=False
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()

        # Save the HTML content to file
        save_html(url, response.text)  # For HTML content, we use response.text

        soup = BeautifulSoup(response.text, 'html.parser')

        # Get and print the page title
        print("Title:", soup.title.string if soup.title else "No Title")

        # Get all <a> tags with href
        for link_tag in soup.find_all('a', href=True):
            full_url = urljoin(url, link_tag['href'])
            if full_url.startswith("http://utm.rnu.tn"):
                crawl(full_url)

            # Check for PDFs or DOCs and save them
            if full_url.endswith(('.pdf', '.doc', '.docx')):
                try:
                    pdf_doc_response = requests.get(full_url, headers=headers, timeout=10, verify=False)
                    pdf_doc_response.raise_for_status()
                    save_html(full_url, pdf_doc_response.content, is_pdf_or_doc=True)  # Save as binary
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download {full_url}: {e}")
                
    except requests.exceptions.RequestException as e:
        print(f"Failed to crawl {url}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl a website and download HTML, PDFs, and DOCs.")
    parser.add_argument("start_url", help="The URL to start crawling from.")

    args = parser.parse_args()
    
    # Start crawling from the provided start URL
    crawl(args.start_url)
