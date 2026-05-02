#!/usr/bin/env python3
"""Stream C3VDv2 dataset directly to LANTA via SSH using a real browser (bypasses Cloudflare).

Resume: already-complete files are skipped based on remote file size check.
Cookie refresh: Playwright re-launches automatically when Cloudflare session expires (403).
"""

import sys
import json
import argparse
import subprocess
import os

from curl_cffi import requests  # Chrome TLS fingerprint — bypasses Cloudflare bot check
from tqdm import tqdm
from playwright.sync_api import sync_playwright

DOI = "10.7281/T1/JC64MK"
BASE_URL = "https://archive.data.jhu.edu"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}


# -----------------------------
# Playwright helpers
# -----------------------------
def launch_browser(playwright, headless, debug=False):
    # Try real Chrome first, fall back to Chromium with stealth args
    for channel in ("chrome", "msedge", None):
        try:
            kwargs = dict(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            if channel:
                kwargs["channel"] = channel
            browser = playwright.chromium.launch(**kwargs)
            if debug:
                print(f"  [debug] launched browser: channel={channel or 'chromium'}")
            break
        except Exception as e:
            if debug:
                print(f"  [debug] channel={channel} unavailable: {e}")
            if channel is None:
                raise
    context = browser.new_context(user_agent=HEADERS["User-Agent"])
    page = context.new_page()
    page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return browser, page


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
        return json.loads(page.inner_text("body"))
    except Exception as e:
        print(f"ERROR: Failed to parse metadata: {e}")
        sys.exit(1)


def get_files(metadata):
    try:
        return metadata["data"]["latestVersion"]["files"]
    except KeyError:
        print("ERROR: Unexpected metadata format")
        sys.exit(1)


def get_browser_cookies(page):
    return {c["name"]: c["value"] for c in page.context.cookies()}


def refresh_cookies(playwright, headless, debug=False):
    """Re-launch browser and return fresh Cloudflare cookies."""
    print("  [session expired] refreshing Cloudflare cookies...", end=" ", flush=True)
    browser, page = launch_browser(playwright, headless, debug=debug)
    url = f"{BASE_URL}/api/datasets/:persistentId?persistentId=doi:{DOI}"
    page.goto(url, wait_until="networkidle", timeout=30000)
    cookies = get_browser_cookies(page)
    browser.close()
    print(f"got {len(cookies)} cookies: {', '.join(cookies.keys()) or 'NONE'}")
    if debug:
        # Test a small request with the new cookies to verify they work
        test_url = f"{BASE_URL}/api/datasets/:persistentId?persistentId=doi:{DOI}"
        r = requests.get(test_url, headers=HEADERS, cookies=cookies, timeout=15, impersonate="chrome")
        print(f"  [debug] test request → HTTP {r.status_code}")
        if r.status_code != 200:
            print(f"  [debug] response headers: {dict(r.headers)}")
    return cookies


# -----------------------------
# Transfer helpers
# -----------------------------
def format_size(bytes_size):
    return f"{bytes_size / 1024 / 1024:.2f} MB"


def get_remote_file_size(ssh_alias, remote_path):
    result = subprocess.run(
        ["ssh", ssh_alias, f"stat -c%s '{remote_path}' 2>/dev/null || echo -1"],
        capture_output=True, text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def transfer_file(file_id, filename, expected_size, ssh_alias, remote_dir, cookies,
                  playwright, headless, debug=False):
    """Stream JHU → SSH pipe → LANTA. Refreshes cookies on 403. Zero local disk."""
    remote_path = f"{remote_dir}/{filename}"
    url = f"{BASE_URL}/api/access/datafile/{file_id}"

    # Resume check
    remote_size = get_remote_file_size(ssh_alias, remote_path)
    if remote_size == expected_size:
        print(f"  Already complete on LANTA, skipping.")
        return True, cookies
    if 0 < remote_size < expected_size:
        print(f"  Resuming from {remote_size / 1024**2:.1f} MB / {expected_size / 1024**2:.1f} MB")

    for attempt in range(3):
        start_byte = max(get_remote_file_size(ssh_alias, remote_path), 0)
        hdrs = {**HEADERS}
        if start_byte > 0:
            hdrs["Range"] = f"bytes={start_byte}-"

        r = requests.get(url, headers=hdrs, cookies=cookies, stream=True, timeout=3600, impersonate="chrome")

        if r.status_code == 403:
            if debug:
                print(f"  [debug] 403 headers: {dict(r.headers)}")
            cookies = refresh_cookies(playwright, headless, debug=debug)
            continue  # retry with fresh cookies

        if r.status_code not in (200, 206):
            print(f"  ERROR: HTTP {r.status_code} for {filename}")
            return False, cookies

        resuming = r.status_code == 206
        start = start_byte if resuming else 0
        ssh_write = f"cat >> '{remote_path}'" if resuming else f"cat > '{remote_path}'"
        ssh_proc = subprocess.Popen(["ssh", ssh_alias, ssh_write], stdin=subprocess.PIPE)

        with tqdm(
            total=expected_size,
            initial=start,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc=filename[:50], ncols=100,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                ssh_proc.stdin.write(chunk)
                pbar.update(len(chunk))

        ssh_proc.stdin.close()
        ssh_proc.wait()

        if ssh_proc.returncode != 0:
            print(f"  ERROR: SSH pipe failed for {filename}")
            return False, cookies

        return True, cookies

    print(f"  ERROR: failed after 3 attempts (persistent 403) for {filename}")
    return False, cookies


def unzip_on_lanta(filename, ssh_alias, remote_dir, delete_zip):
    remote_path = f"{remote_dir}/{filename}"
    extract_dir = f"{remote_dir}/{filename[:-4]}"
    print(f"  Extracting on LANTA...", end=" ", flush=True)
    cmd = f"unzip -o '{remote_path}' -d '{extract_dir}'"
    if delete_zip:
        cmd += f" && rm '{remote_path}'"
    result = subprocess.run(["ssh", ssh_alias, cmd], capture_output=True)
    if result.returncode != 0:
        print(f"FAILED\n  {result.stderr.decode().strip()}")
    else:
        print("done.")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_dir", help="Target directory on LANTA")
    parser.add_argument("--ssh-alias", default="lanta", help="SSH alias from ~/.ssh/config")
    parser.add_argument("--no-unzip", action="store_true", help="Skip unzipping on LANTA after transfer")
    parser.add_argument("--delete-zip", action="store_true", help="Delete zip on LANTA after extracting")
    parser.add_argument("--no-headless", dest="headless", action="store_false", default=True,
                        help="Show browser window")
    parser.add_argument("--debug", action="store_true",
                        help="Print browser/cookie/HTTP debug info")
    args = parser.parse_args()

    print(f"SSH alias:  {args.ssh_alias}")
    print(f"Remote dir: {args.remote_dir}")
    print(f"Auto-unzip: {not args.no_unzip}")
    print()

    with sync_playwright() as playwright:
        browser, page = launch_browser(playwright, args.headless, debug=args.debug)

        print("Fetching metadata...")
        metadata = fetch_metadata_via_api(page)
        files = get_files(metadata)
        cookies = get_browser_cookies(page)
        if args.debug:
            print(f"  [debug] initial cookies: {', '.join(cookies.keys()) or 'NONE'}")
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

            # Skip if already extracted (zip was deleted after a previous run)
            if filename.endswith(".zip") and not args.no_unzip:
                extract_dir = f"{args.remote_dir}/{filename[:-4]}"
                check = subprocess.run(
                    ["ssh", args.ssh_alias, f"test -d '{extract_dir}' && echo exists || echo missing"],
                    capture_output=True, text=True,
                )
                if check.stdout.strip() == "exists":
                    print(f"  Already extracted on LANTA, skipping.")
                    continue

            ok, cookies = transfer_file(
                file_id=file_id,
                filename=filename,
                expected_size=expected_size,
                ssh_alias=args.ssh_alias,
                remote_dir=args.remote_dir,
                cookies=cookies,
                playwright=playwright,
                headless=args.headless,
                debug=args.debug,
            )
            if not ok:
                failed.append(filename)
                continue

            if not args.no_unzip and filename.endswith(".zip"):
                unzip_on_lanta(filename, args.ssh_alias, args.remote_dir, args.delete_zip)

    print("\n" + "=" * 100)
    print(f"Done. {total - len(failed)}/{total} files transferred successfully.")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
