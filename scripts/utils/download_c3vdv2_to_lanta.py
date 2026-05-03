#!/usr/bin/env python3
"""Stream C3VDv2 dataset directly to LANTA via SSH using a real browser (bypasses Cloudflare).

Resume: already-complete files are skipped based on remote file size check.
Cookie refresh: Playwright re-launches automatically when Cloudflare session expires (403).
Parallel: use --parallel N to download N files simultaneously.
"""

import sys
import json
import argparse
import subprocess
import os
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from curl_cffi import requests  # Chrome TLS fingerprint — bypasses Cloudflare bot check
from tqdm import tqdm
from playwright.sync_api import sync_playwright

DOI = "10.7281/T1/JC64MK"
BASE_URL = "https://archive.data.jhu.edu"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB — larger chunks reduce syscall overhead

# SSH ControlMaster multiplexing (Unix only — not supported by Windows OpenSSH)
_IS_WINDOWS = sys.platform == "win32"
_KEEPALIVE_OPTS = [
    "-o", "ServerAliveInterval=15",   # send keepalive every 15s so idle pipe doesn't drop
    "-o", "ServerAliveCountMax=4",
    "-o", "Compression=no",           # zip files don't compress further; saves CPU
    "-o", "StrictHostKeyChecking=no",
]

if not _IS_WINDOWS:
    _SSH_CTRL = os.path.join(tempfile.gettempdir(), "lanta_dl_%C")
    SSH_OPTS = [
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={_SSH_CTRL}",
        "-o", "ControlPersist=300",
        *_KEEPALIVE_OPTS,
    ]
else:
    # Windows OpenSSH doesn't support ControlMaster
    SSH_OPTS = _KEEPALIVE_OPTS


def ssh_run(ssh_alias, remote_cmd, **kwargs):
    return subprocess.run(["ssh"] + SSH_OPTS + [ssh_alias, remote_cmd], **kwargs)


def ssh_popen(ssh_alias, remote_cmd, **kwargs):
    return subprocess.Popen(["ssh"] + SSH_OPTS + [ssh_alias, remote_cmd], **kwargs)


# -----------------------------
# Playwright helpers
# -----------------------------
def launch_browser(playwright, headless, debug=False):
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


# -----------------------------
# Thread-safe cookie manager
# -----------------------------
class CookieManager:
    """Shared cookies with thread-safe 403-triggered refresh (only one refresh at a time)."""

    def __init__(self, cookies, playwright, headless, debug=False):
        self._cookies = cookies
        self._playwright = playwright
        self._headless = headless
        self._debug = debug
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return dict(self._cookies)

    def refresh(self, stale_cookies):
        """Refresh only if cookies haven't been updated by another thread already."""
        with self._lock:
            if self._cookies == stale_cookies:
                print("  [session expired] refreshing Cloudflare cookies...", end=" ", flush=True)
                url = f"{BASE_URL}/api/datasets/:persistentId?persistentId=doi:{DOI}"
                browser, page = launch_browser(self._playwright, self._headless, debug=self._debug)
                page.goto(url, wait_until="networkidle", timeout=30000)
                self._cookies = get_browser_cookies(page)
                browser.close()
                print(f"got {len(self._cookies)} cookies.")
            return dict(self._cookies)


# -----------------------------
# Transfer helpers
# -----------------------------
def format_size(bytes_size):
    return f"{bytes_size / 1024 / 1024:.2f} MB"


def batch_check_remote(ssh_alias, remote_dir, files, check_extract=True):
    """Single SSH call to get every file's remote size and (optionally) extract-dir existence.

    Returns dict: filename -> {"size": int, "extracted": bool}
    """
    parts = []
    for f in files:
        fn = f["label"]
        rp = f"{remote_dir}/{fn}"
        ed = f"{remote_dir}/{fn[:-4]}"
        # Print "filename|size|extracted" — all in one shell session
        parts.append(
            f"sz=$(stat -c%s '{rp}' 2>/dev/null || echo -1); "
            + (f"ex=$([ -d '{ed}' ] && echo 1 || echo 0); " if check_extract else "ex=0; ")
            + f"echo '{fn}|'$sz'|'$ex"
        )
    script = "; ".join(parts)
    result = ssh_run(ssh_alias, script, capture_output=True, text=True)
    info = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts_line = line.split("|")
        if len(parts_line) < 3:
            continue
        fn = parts_line[0]
        try:
            size = int(parts_line[1])
        except ValueError:
            size = -1
        extracted = parts_line[2] == "1"
        info[fn] = {"size": size, "extracted": extracted}
    return info


def transfer_file(file_id, filename, expected_size, ssh_alias, remote_dir,
                  cookie_mgr, remote_size=-1, tqdm_pos=0, debug=False):
    """Stream JHU → SSH pipe → LANTA. Refreshes cookies on 403. Zero local disk."""
    remote_path = f"{remote_dir}/{filename}"
    url = f"{BASE_URL}/api/access/datafile/{file_id}"

    if remote_size == expected_size:
        tqdm.write(f"  [{filename}] Already complete on LANTA, skipping.")
        return True

    if 0 < remote_size < expected_size:
        tqdm.write(f"  [{filename}] Resuming from {remote_size / 1024**2:.1f} MB / {expected_size / 1024**2:.1f} MB")

    cookies = cookie_mgr.get()
    start_byte = max(remote_size, 0)

    for attempt in range(3):
        hdrs = {**HEADERS}
        if start_byte > 0:
            hdrs["Range"] = f"bytes={start_byte}-"

        r = requests.get(url, headers=hdrs, cookies=cookies, stream=True, timeout=3600, impersonate="chrome")

        if r.status_code == 403:
            if debug:
                tqdm.write(f"  [debug] 403 headers: {dict(r.headers)}")
            cookies = cookie_mgr.refresh(cookies)
            continue

        if r.status_code not in (200, 206):
            tqdm.write(f"  ERROR: HTTP {r.status_code} for {filename}")
            return False

        resuming = r.status_code == 206
        start = start_byte if resuming else 0
        ssh_write = f"cat >> '{remote_path}'" if resuming else f"cat > '{remote_path}'"
        ssh_proc = ssh_popen(ssh_alias, ssh_write, stdin=subprocess.PIPE)

        with tqdm(
            total=expected_size,
            initial=start,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc=filename[:40],
            ncols=100,
            position=tqdm_pos,
            leave=True,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                ssh_proc.stdin.write(chunk)
                pbar.update(len(chunk))

        ssh_proc.stdin.close()
        ssh_proc.wait()

        if ssh_proc.returncode != 0:
            tqdm.write(f"  ERROR: SSH pipe failed for {filename}")
            return False

        return True

    tqdm.write(f"  ERROR: failed after 3 attempts (persistent 403) for {filename}")
    return False


def unzip_on_lanta(filename, ssh_alias, remote_dir, delete_zip):
    remote_path = f"{remote_dir}/{filename}"
    extract_dir = f"{remote_dir}/{filename[:-4]}"
    tqdm.write(f"  [{filename}] Extracting on LANTA...", end="")
    cmd = f"unzip -o '{remote_path}' -d '{extract_dir}'"
    if delete_zip:
        cmd += f" && rm '{remote_path}'"
    result = ssh_run(ssh_alias, cmd, capture_output=True)
    if result.returncode != 0:
        tqdm.write(f"FAILED\n  {result.stderr.decode().strip()}")
    else:
        tqdm.write(" done.")


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
    parser.add_argument("--parallel", type=int, default=1, metavar="N",
                        help="Number of simultaneous file transfers (default: 1; increase only if JHU allows it)")
    args = parser.parse_args()

    print(f"SSH alias:  {args.ssh_alias}")
    print(f"Remote dir: {args.remote_dir}")
    print(f"Auto-unzip: {not args.no_unzip}")
    print(f"Parallel:   {args.parallel} concurrent transfers")
    print()

    with sync_playwright() as playwright:
        browser, page = launch_browser(playwright, args.headless, debug=args.debug)

        print("Fetching metadata...")
        metadata = fetch_metadata_via_api(page)
        files = get_files(metadata)
        initial_cookies = get_browser_cookies(page)
        if args.debug:
            print(f"  [debug] initial cookies: {', '.join(initial_cookies.keys()) or 'NONE'}")
        browser.close()

        cookie_mgr = CookieManager(initial_cookies, playwright, args.headless, debug=args.debug)

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
        ssh_run(args.ssh_alias, f"mkdir -p '{args.remote_dir}'", check=True)

        selected_files = [files[i] for i in selected]

        print(f"Checking remote state for {len(selected_files)} files (1 SSH call)...")
        remote_info = batch_check_remote(
            args.ssh_alias, args.remote_dir, selected_files,
            check_extract=not args.no_unzip,
        )

        # Filter out already-done files
        work_items = []
        for f in selected_files:
            filename = f["label"]
            info = remote_info.get(filename, {"size": -1, "extracted": False})
            if filename.endswith(".zip") and not args.no_unzip and info["extracted"]:
                print(f"  [{filename}] Already extracted on LANTA, skipping.")
                continue
            work_items.append((f, info["size"]))

        total = len(work_items)
        print(f"\nStarting transfers ({total} files, {args.parallel} parallel)...\n")

        failed = []
        unzip_queue = []
        lock = threading.Lock()

        def worker(f, known_remote_size, pos):
            file_id = f["dataFile"]["id"]
            filename = f["label"]
            expected_size = f["dataFile"]["filesize"]
            tqdm.write(f"  → {filename} ({format_size(expected_size)})")
            ok = transfer_file(
                file_id=file_id,
                filename=filename,
                expected_size=expected_size,
                ssh_alias=args.ssh_alias,
                remote_dir=args.remote_dir,
                cookie_mgr=cookie_mgr,
                remote_size=known_remote_size,
                tqdm_pos=pos,
                debug=args.debug,
            )
            if ok and not args.no_unzip and filename.endswith(".zip"):
                with lock:
                    unzip_queue.append(filename)
            return filename, ok

        # Use a pool of tqdm positions (one bar per parallel slot)
        positions = list(range(args.parallel))
        pos_lock = threading.Lock()
        available_pos = list(positions)

        def worker_with_pos(item):
            f, known_remote_size = item
            with pos_lock:
                pos = available_pos.pop(0)
            try:
                return worker(f, known_remote_size, pos)
            finally:
                with pos_lock:
                    available_pos.append(pos)

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(worker_with_pos, item): item for item in work_items}
            for fut in as_completed(futures):
                filename, ok = fut.result()
                if not ok:
                    failed.append(filename)

        # Unzip sequentially after all downloads (avoids overloading LANTA I/O)
        for filename in unzip_queue:
            unzip_on_lanta(filename, args.ssh_alias, args.remote_dir, args.delete_zip)

    print("\n" + "=" * 100)
    print(f"Done. {total - len(failed)}/{total} files transferred successfully.")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
