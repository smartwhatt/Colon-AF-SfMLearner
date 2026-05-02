#!/usr/bin/env python3
"""Download C3VDv2 dataset to local machine using a real browser (bypasses Cloudflare)."""

import os
import sys
import json
import argparse
import zipfile

from playwright.sync_api import sync_playwright

DOI = "10.7281/T1/JC64MK"
BASE_URL = "https://archive.data.jhu.edu"


def fetch_metadata(page):
    url = f"{BASE_URL}/api/datasets/:persistentId?persistentId=doi:{DOI}"
    response = page.goto(url, wait_until="networkidle")
    if response.status != 200:
        print(f"ERROR: metadata fetch returned HTTP {response.status}")
        sys.exit(1)
    return json.loads(page.content().split("<body>")[1].split("</body>")[0]
                      if "<body>" in page.content() else page.inner_text("pre") or page.content())


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

    # fallback: read text from page
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
        print(json.dumps(metadata, indent=2)[:500])
        sys.exit(1)


def format_size(bytes_size):
    return f"{bytes_size / 1024 / 1024:.2f} MB"


def download_file(page, file_id, filename, output_dir):
    url = f"{BASE_URL}/api/access/datafile/{file_id}"
    output_path = os.path.join(output_dir, filename)

    print(f"  Downloading {filename}...", end=" ", flush=True)

    with page.expect_download() as dl_info:
        page.evaluate(f"""
            const a = document.createElement('a');
            a.href = '{url}';
            a.download = '{filename}';
            document.body.appendChild(a);
            a.click();
        """)

    download = dl_info.value
    download.save_as(output_path)
    print("done.")
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
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run browser headless (default: True)")
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Show browser window")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Download directory: {args.output_dir}")
    print(f"Auto-unzip: {not args.no_unzip}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            accept_downloads=True,
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )
        page = context.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        print("Fetching metadata...")
        metadata = fetch_metadata_via_api(page)
        files = get_files(metadata)

        total_size = sum(f["dataFile"]["filesize"] for f in files)

        print("\nAvailable files:")
        print("-" * 100)
        for i, f in enumerate(files):
            name = f["label"]
            size = format_size(f["dataFile"]["filesize"])
            print(f"[{i:3d}] {name:<60} {size}")
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

        # Navigate to dataset page first so cookies/session are set
        page.goto(f"{BASE_URL}/dataset.xhtml?persistentId=doi:{DOI}", wait_until="networkidle")

        print("\nStarting downloads...")
        print("-" * 100)

        for i in selected:
            f = files[i]
            file_id = f["dataFile"]["id"]
            filename = f["label"]

            output_path = download_file(page, file_id, filename, args.output_dir)

            if not args.no_unzip and filename.endswith(".zip"):
                extract_dir = os.path.join(args.output_dir, filename.replace(".zip", ""))
                os.makedirs(extract_dir, exist_ok=True)
                print(f"  Extracting {filename}...")
                unzip_file(output_path, extract_dir, args.delete_zip)

        print("-" * 100)
        print("Done.")
        browser.close()


if __name__ == "__main__":
    main()
