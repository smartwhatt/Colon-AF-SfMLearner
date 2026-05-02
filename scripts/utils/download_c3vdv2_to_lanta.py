#!/usr/bin/env python3
"""Stream C3VDv2 dataset directly to LANTA via SSH using a real browser (bypasses Cloudflare).

Resume: already-complete files are skipped based on remote file size check.
"""

import sys
import json
import argparse
import subprocess
import tempfile
import os

import requests
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
    """Extract Playwright session cookies for use with requests."""
    cookies = page.context.cookies()
    return {c["name"]: c["value"] for c in cookies}


def get_remote_file_size(ssh_alias, remote_path):
    """Return file size on LANTA, or -1 if file doesn't exist."""
    result = subprocess.run(
        ["ssh", ssh_alias, f"stat -c%s '{remote_path}' 2>/dev/null || echo -1"],
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def transfer_file(file_id, filename, expected_size, ssh_alias, remote_dir, cookies):
    remote_path = f"{remote_dir}/{filename}"
    url = f"{BASE_URL}/api/access/datafile/{file_id}"

    # Resume check
    remote_size = get_remote_file_size(ssh_alias, remote_path)
    if remote_size == expected_size:
        print(f"  [{filename}] already complete on LANTA, skipping.")
        return True
    if remote_size > 0:
        print(f"  [{filename}] partial file on LANTA ({remote_size / 1024**2:.1f} MB / {expected_size / 1024**2:.1f} MB), resuming...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }
    start_byte = max(remote_size, 0)
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, filename)

        # Download to temp file with progress bar
        with requests.get(url, headers=headers, cookies=cookies, stream=True, timeout=3600) as r:
            if r.status_code not in (200, 206):
                print(f"  ERROR: HTTP {r.status_code} for {filename}")
                return False

            resuming = r.status_code == 206
            downloaded_bytes = start_byte if resuming else 0
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
                with open(tmp_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Transfer to LANTA
        print(f"  Transferring to LANTA...", end=" ", flush=True)
        scp_result = subprocess.run(
            ["scp", tmp_path, f"{ssh_alias}:{remote_path}"],
            capture_output=True,
        )
        if scp_result.returncode != 0:
            print(f"FAILED\n  scp error: {scp_result.stderr.decode()}")
            return False

        print("done.")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_dir", help="Target directory on LANTA")
    parser.add_argument("--ssh-alias", default="lanta", help="SSH alias from ~/.ssh/config")
    parser.add_argument("--no-headless", dest="headless", action="store_false", default=True,
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

    print("\nCreating remote directory...")
    subprocess.run(["ssh", args.ssh_alias, f"mkdir -p '{args.remote_dir}'"], check=True)

    total = len(selected)
    print(f"\nStarting transfers ({total} files)...\n")

    failed = []
    for n, i in enumerate(selected, 1):
        f = files[i]
        file_id = f["dataFile"]["id"]
        filename = f["label"]
        expected_size = f["dataFile"]["filesize"]

        print(f"[{n}/{total}] {filename} ({format_size(expected_size)})")
        ok = transfer_file(
            file_id=file_id,
            filename=filename,
            expected_size=expected_size,
            ssh_alias=args.ssh_alias,
            remote_dir=args.remote_dir,
            cookies=cookies,
        )
        if not ok:
            failed.append(filename)

    print("\n" + "=" * 100)
    print(f"Done. {total - len(failed)}/{total} files transferred successfully.")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
