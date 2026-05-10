# helpers/download_ontology.py

import argparse
import re
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests


HEADERS = {
    "User-Agent": "Mozilla/5.0 ontology-downloader/1.0",
    "Accept": (
        "application/rdf+xml, application/owl+xml, application/xml, "
        "text/turtle, application/n-triples, application/zip, */*"
    ),
}


def filename_from_response(response: requests.Response, url: str) -> str:
    cd = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename="?([^";]+)"?', cd)
    if match:
        return match.group(1)

    name = Path(urlparse(response.url).path).name or Path(urlparse(url).path).name
    return name or "ontology_download.owl"


def download_ontology(
    url: str,
    destination_folder: str,
    filename: str | None = None,
    extract: bool = False,
) -> Path:
    dest_dir = Path(destination_folder).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Requesting: {url}")

    with requests.get(
        url,
        headers=HEADERS,
        stream=True,
        allow_redirects=True,
        timeout=(10, 30),  # 10 sec connect, 30 sec between data chunks
    ) as response:
        print(f"Final URL: {response.url}")
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'unknown')}")
        print(f"Content-Length: {response.headers.get('Content-Length', 'unknown')}")

        response.raise_for_status()

        output_name = filename or filename_from_response(response, url)
        output_path = dest_dir / output_name

        print(f"Saving to: {output_path}")

        bytes_written = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                f.write(chunk)
                bytes_written += len(chunk)
                print(f"Downloaded {bytes_written:,} bytes", end="\r")

        print(f"\nDownload complete: {bytes_written:,} bytes")

    if bytes_written == 0:
        raise RuntimeError("Download completed but zero bytes were written.")

    if extract and output_path.suffix.lower() == ".zip":
        extract_dir = dest_dir / output_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting to: {extract_dir}")

        with zipfile.ZipFile(output_path, "r") as z:
            z.extractall(extract_dir)

        print("Extraction complete.")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("destination_folder")
    parser.add_argument("--filename", default=None)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    download_ontology(
        url=args.url,
        destination_folder=args.destination_folder,
        filename=args.filename,
        extract=args.extract,
    )


if __name__ == "__main__":
    main()