#!/bin/bash

set -e

DOI="10.7281/T1/JC64MK"
BASE_URL="https://archive.data.jhu.edu"

# ---- CLI ARGUMENTS ----
DOWNLOAD_DIR=${1:-./downloads}
UNZIP=${2:-true}        # true / false
DELETE_ZIP=${3:-false}  # true / false

mkdir -p "$DOWNLOAD_DIR"

echo "Download directory: $DOWNLOAD_DIR"
echo "Auto-unzip: $UNZIP"
echo "Delete zip after unzip: $DELETE_ZIP"
echo ""

# ---- FETCH METADATA ----
echo "Fetching dataset metadata..."
metadata=$(curl -s "$BASE_URL/api/datasets/:persistentId?persistentId=doi:$DOI")

if [ -z "$metadata" ]; then
    echo "Failed to fetch dataset metadata"
    exit 1
fi

# ---- PARSE FILES ----
files=$(python3 - <<EOF
import json, sys

data = json.load(sys.stdin)

for f in data["data"]["latestVersion"]["files"]:
    file_id = f["dataFile"]["id"]
    name = f["label"]
    size = f["dataFile"]["filesize"]
    print(f"{file_id}|{name}|{size}")
EOF
<<< "$metadata")

total_size=0
index=0

declare -a ids
declare -a names
declare -a sizes

echo "Available files:"
echo "---------------------------------------------"

while IFS="|" read -r id name size; do
    size_mb=$(awk "BEGIN {printf \"%.2f\", $size/1024/1024}")
    echo "[$index] $name (${size_mb} MB)"

    ids[$index]=$id
    names[$index]=$name
    sizes[$index]=$size

    total_size=$((total_size + size))
    index=$((index + 1))
done <<< "$files"

total_size_gb=$(awk "BEGIN {printf \"%.2f\", $total_size/1024/1024/1024}")
echo "---------------------------------------------"
echo "Total dataset size: ${total_size_gb} GB"
echo ""

# ---- USER SELECTION ----
echo "Options:"
echo "1) Download ALL files"
echo "2) Download first N files (small sample)"
echo "3) Select files manually"
read -p "Choose option [1-3]: " choice

selected_indices=()

if [ "$choice" == "1" ]; then
    for i in "${!ids[@]}"; do
        selected_indices+=($i)
    done

elif [ "$choice" == "2" ]; then
    read -p "Enter N: " N
    for ((i=0; i<N && i<${#ids[@]}; i++)); do
        selected_indices+=($i)
    done

elif [ "$choice" == "3" ]; then
    read -p "Enter indices (space separated): " input
    selected_indices=($input)
else
    echo "Invalid choice"
    exit 1
fi

# ---- DOWNLOAD + UNZIP ----
echo ""
echo "Starting download..."
echo "---------------------------------------------"

for i in "${selected_indices[@]}"; do
    file_id=${ids[$i]}
    filename=${names[$i]}
    filepath="$DOWNLOAD_DIR/$filename"

    echo "Downloading $filename..."

    download_url=$(curl -s -D - -o /dev/null \
        "$BASE_URL/api/access/datafile/$file_id" \
        | grep -i '^Location:' | cut -d' ' -f2 | tr -d '\r')

    curl -L -o "$filepath" "$download_url"

    # ---- UNZIP ----
    if [ "$UNZIP" == "true" ] && [[ "$filename" == *.zip ]]; then
        echo "Extracting $filename..."

        extract_dir="$DOWNLOAD_DIR/${filename%.zip}"
        mkdir -p "$extract_dir"

        unzip -q "$filepath" -d "$extract_dir"

        if [ "$DELETE_ZIP" == "true" ]; then
            rm "$filepath"
            echo "Deleted $filename"
        fi
    fi

    echo ""
done

echo "---------------------------------------------"
echo "All done."