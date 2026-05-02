#!/usr/bin/env python3
"""Stream C3VDv2 dataset directly to LANTA via SSH using a real browser (bypasses Cloudflare)."""

import sys
import json
import argparse
import subprocess
import tempfile
import os

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


def stream_to_lanta(page, file_id, filename, ssh_alias, remote_dir):
    print(f"\nTransferring {filename} → {ssh_alias}:{remote_dir}")

    url = f"{BASE_URL}/api/access/datafile/{file_id}"

    # Download to a temp file via browser, then scp to LANTA
    with tempfile.TemporaryDirectory() as tmpdir:
        context = page.context
        # Use a fresh page with downloads enabled to grab the file
        dl_page = context.new_page()

        with dl_page.expect_download(timeout=3600000) as dl_info:
            dl_page.evaluate(f"""
                const a = document.createElement('a');
                a.href = '{url}';
                a.download = '{filename}';
                document.body.appendChild(a);
                a.click();
            """)

        download = dl_info.value
        tmp_path = os.path.join(tmpdir, filename)
        download.save_as(tmp_path)
        dl_page.close()

        print(f"  Downloaded locally, transferring to LANTA...", end=" ", flush=True)
        result = subprocess.run(
            ["scp", tmp_path, f"{ssh_alias}:{remote_dir}/{filename}"],
            capture_output=True,
        )

        if result.returncode != 0:
            print(f"FAILED\n  scp error: {result.stderr.decode()}")
            return False

        print("done.")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_dir", help="Target directory on LANTA")
    parser.add_argument("--ssh-alias", default="lanta", help="SSH alias from ~/.ssh/config")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Show browser window")
    args = parser.parse_args()

    print(f"SSH alias: {args.ssh_alias}")
    print(f"Remote dir: {args.remote_dir}")
    print()

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
        print(f"Total dataset size: {total_size / 1024 / 1024 / 1024:.2f} GB")

        print("\nOptions:")
        print("1) Download ALL files")
        print("2) Download first N files")
        print("3) Select manually")

        choice = input("\nChoose option [1-3]: ").strip()
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

        # Visit dataset page to warm up session/cookies
        page.goto(f"{BASE_URL}/dataset.xhtml?persistentId=doi:{DOI}", wait_until="networkidle")

        print("\nCreating remote directory...")
        subprocess.run(["ssh", args.ssh_alias, f"mkdir -p '{args.remote_dir}'"], check=True)

        print("\nStarting transfers...")
        for i in selected:
            f = files[i]
            stream_to_lanta(
                page=page,
                file_id=f["dataFile"]["id"],
                filename=f["label"],
                ssh_alias=args.ssh_alias,
                remote_dir=args.remote_dir,
            )

        print("\nDone.")
        browser.close()


if __name__ == "__main__":
    main()
