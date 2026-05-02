#!/usr/bin/env python3
"""Download C3VDv2 dataset to local machine using a real browser (bypasses Cloudflare).

Resume: already-complete files are skipped based on local file size check.
"""

import os
import sys
import json
import argparse
import zipfile

from curl_cffi import requests  # Chrome TLS fingerprint — bypasses Cloudflare bot check
from tqdm import tqdm
from playwright.sync_api import sync_playwright

DOI = "10.7281/T1/JC64MK"
BASE_URL = "https://archive.data.jhu.edu"


def fetch_metadata_via_api(page):
    url = f"{BASE_URL}/api/datasets/:persistentId?persistentId=doi:{DOI}"
    result = {}

    def handle_response(response):
        if ":persistentId" in response.url and response.status == 200:
            try:
                result["data"] = response.json()
            except Exception:
                pass

    page.on("response", handle_response)
    page.goto(url, wait_until="networkidle", timeout=30000)

    if "data" in result:
        return result["data"]

    try:
        text = page.inner_text("body")
        return json.loads(text)
    except Exception as e:
        print(f"ERROR: Failed to parse metadata: {e}")
        sys.exit(1)


def get_files(metadata):
    try:
        return metadata["data"]["latestVersion"]["files"]
    except KeyError:
        print("ERROR: Unexpected metadata format")
        sys.exit(1)


def format_size(bytes_size):
    return f"{bytes_size / 1024 / 1024:.2f} MB"


def get_browser_cookies(page):
    cookies = page.context.cookies()
    return {c["name"]: c["value"] for c in cookies}


def download_file(file_id, filename, expected_size, output_dir, cookies):
    output_path = os.path.join(output_dir, filename)
    url = f"{BASE_URL}/api/access/datafile/{file_id}"

    # Resume check
    existing_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    if existing_size == expected_size:
        print(f"  [{filename}] already complete, skipping.")
        return output_path
    if existing_size > 0:
        print(f"  [{filename}] partial file ({existing_size / 1024**2:.1f} MB / {expected_size / 1024**2:.1f} MB), resuming...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    with requests.get(url, headers=headers, cookies=cookies, stream=True, timeout=3600, impersonate="chrome") as r:
        if r.status_code not in (200, 206):
            print(f"  ERROR: HTTP {r.status_code} for {filename}")
            return None

        resuming = r.status_code == 206
        downloaded_bytes = existing_size if resuming else 0
        mode = "ab" if resuming else "wb"

        with tqdm(
            total=expected_size,
            initial=downloaded_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename[:50],
            ncols=100,
        ) as pbar:
            with open(output_path, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    pbar.update(len(chunk))

    return output_path


def unzip_file(zip_path, extract_dir, delete_zip):
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        if delete_zip:
            os.remove(zip_path)
    except Exception as e:
        print(f"Unzip failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="./downloads")
    parser.add_argument("--no-unzip", action="store_true")
    parser.add_argument("--delete-zip", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false", default=True,
                        help="Show browser window")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Download directory: {args.output_dir}")
    print(f"Auto-unzip: {not args.no_unzip}")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )
        page = context.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        print("Fetching metadata...")
        metadata = fetch_metadata_via_api(page)
        files = get_files(metadata)
        cookies = get_browser_cookies(page)
        browser.close()

    total_size = sum(f["dataFile"]["filesize"] for f in files)

    print("\nAvailable files:")
    print("-" * 100)
    for i, f in enumerate(files):
        print(f"[{i:3d}] {f['label']:<60} {format_size(f['dataFile']['filesize'])}")
    print("-" * 100)
    print(f"Total dataset size: {total_size / 1024 / 1024 / 1024:.2f} GB\n")

    print("Options:")
    print("1) Download ALL files")
    print("2) Download first N files")
    print("3) Select manually")

    choice = input("Choose option [1-3]: ").strip()
    selected = []

    if choice == "1":
        selected = list(range(len(files)))
    elif choice == "2":
        N = int(input("Enter N: "))
        selected = list(range(min(N, len(files))))
    elif choice == "3":
        indices = input("Enter indices (space separated): ").split()
        selected = [int(i) for i in indices]
    else:
        print("Invalid choice")
        sys.exit(1)

    total = len(selected)
    print(f"\nStarting downloads ({total} files)...\n")

    failed = []
    for n, i in enumerate(selected, 1):
        f = files[i]
        file_id = f["dataFile"]["id"]
        filename = f["label"]
        expected_size = f["dataFile"]["filesize"]

        print(f"[{n}/{total}] {filename} ({format_size(expected_size)})")
        output_path = download_file(
            file_id=file_id,
            filename=filename,
            expected_size=expected_size,
            output_dir=args.output_dir,
            cookies=cookies,
        )

        if output_path is None:
            failed.append(filename)
            continue

        if not args.no_unzip and filename.endswith(".zip"):
            extract_dir = os.path.join(args.output_dir, filename.replace(".zip", ""))
            os.makedirs(extract_dir, exist_ok=True)
            print(f"  Extracting...")
            unzip_file(output_path, extract_dir, args.delete_zip)

    print("\n" + "=" * 100)
    print(f"Done. {total - len(failed)}/{total} files downloaded successfully.")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
