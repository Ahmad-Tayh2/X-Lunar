import os
import requests
from urllib.parse import urlparse, unquote
from docx2pdf import convert

def download_and_convert_files(urls, output_folder="downloads"):
    os.makedirs(output_folder, exist_ok=True)
    pdf_folder = os.path.join(output_folder, "pdfs")
    os.makedirs(pdf_folder, exist_ok=True)

    for link in urls:
        try:
            filename = os.path.basename(urlparse(link).path)
            filename = unquote(filename)  # Decode URL-encoded characters
            file_path = os.path.join(output_folder, filename)

            print(f"Downloading: {link}")
            response = requests.get(link)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved to: {file_path}")

            # Handle conversion
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".docx":
                convert_to_pdf(file_path, pdf_folder)
            elif ext == ".pdf":
                print(f"{filename} is already a PDF. Skipping conversion.")
            else:
                print(f"{filename} is not a DOCX or PDF. Left as-is.")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {link}: {e}")

def convert_to_pdf(file_path, pdf_output_folder):
    try:
        print(f"Converting to PDF: {file_path}")
        convert(file_path, os.path.join(pdf_output_folder, os.path.basename(file_path) + ".pdf"))
        print(f"PDF saved in: {pdf_output_folder}")
    except Exception as e:
        print(f"Failed to convert {file_path} to PDF: {e}")

# 🔧 Replace this with your list of document links (PDF or DOCX)
doc_urls = [
    "https://utm.rnu.tn/utm/fr/",


]

download_and_convert_files(doc_urls)
