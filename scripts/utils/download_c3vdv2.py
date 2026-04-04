#!/usr/bin/env python3

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
import shutil
import zipfile

DOI = "10.7281/T1/JC64MK"
BASE_URL = "https://archive.data.jhu.edu"


# -----------------------------
# Helpers
# -----------------------------
def fetch_metadata():
    url = f"{BASE_URL}/api/datasets/:persistentId?persistentId=doi:{DOI}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(req) as response:
            data = response.read()
            return json.loads(data)
    except Exception as e:
        print(f"ERROR: Failed to fetch metadata: {e}")
        sys.exit(1)


def get_files(metadata):
    try:
        return metadata["data"]["latestVersion"]["files"]
    except KeyError:
        print("ERROR: Unexpected metadata format")
        sys.exit(1)


def format_size(bytes_size):
    return f"{bytes_size / 1024 / 1024:.2f} MB"


def download_file(file_id, output_path):
    try:
        url = f"{BASE_URL}/api/access/datafile/{file_id}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req) as response, open(output_path, "wb") as out:
            shutil.copyfileobj(response, out)

    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e}")
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        return False

    return True


def unzip_file(zip_path, extract_dir, delete_zip):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        if delete_zip:
            os.remove(zip_path)
    except Exception as e:
        print(f"Unzip failed: {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="./downloads")
    parser.add_argument("--no-unzip", action="store_true")
    parser.add_argument("--delete-zip", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Download directory: {args.output_dir}")
    print(f"Auto-unzip: {not args.no_unzip}")
    print(f"Delete zip: {args.delete_zip}")
    print()

    print("Fetching metadata...")
    metadata = fetch_metadata()
    files = get_files(metadata)

    total_size = sum(f["dataFile"]["filesize"] for f in files)

    print("\nAvailable files:")
    print("---------------------------------------------")

    for i, f in enumerate(files):
        name = f["label"]
        size = format_size(f["dataFile"]["filesize"])
        print(f"[{i}] {name} ({size})")

    print("---------------------------------------------")
    print(f"Total dataset size: {total_size / 1024 / 1024 / 1024:.2f} GB\n")

    # -----------------------------
    # Selection
    # -----------------------------
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

    # -----------------------------
    # Download loop
    # -----------------------------
    print("\nStarting downloads...")
    print("---------------------------------------------")

    for i in selected:
        f = files[i]
        file_id = f["dataFile"]["id"]
        filename = f["label"]

        output_path = os.path.join(args.output_dir, filename)

        print(f"Downloading {filename}...")

        success = download_file(file_id, output_path)
        if not success:
            continue

        # unzip
        if not args.no_unzip and filename.endswith(".zip"):
            extract_dir = os.path.join(args.output_dir, filename.replace(".zip", ""))
            os.makedirs(extract_dir, exist_ok=True)

            print(f"Extracting {filename}...")
            unzip_file(output_path, extract_dir, args.delete_zip)

    print("---------------------------------------------")
    print("Done.")


if __name__ == "__main__":
    main()