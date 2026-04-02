#!/bin/bash

DOI="10.7281/T1/JC64MK"
BASE_URL="https://archive.data.jhu.edu"

echo "Fetching dataset metadata..."
metadata=$(curl -s "$BASE_URL/api/datasets/:persistentId?persistentId=doi:$DOI")

if [ -z "$metadata" ]; then
    echo "Failed to fetch dataset metadata"
    exit 1
fi

# Extract file info: name + size + id
echo ""
echo "Available files:"
echo "---------------------------------------------"

files=$(echo "$metadata" | jq -r '
.data.latestVersion.files[] |
"\(.dataFile.id)|\(.label)|\(.dataFile.filesize)"')

total_size=0
index=0

declare -a ids
declare -a names
declare -a sizes

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

# User options
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
    read -p "Enter N (number of files): " N
    for ((i=0; i<N && i<${#ids[@]}; i++)); do
        selected_indices+=($i)
    done

elif [ "$choice" == "3" ]; then
    read -p "Enter indices (space separated, e.g., 0 2 5): " input
    selected_indices=($input)
else
    echo "Invalid choice"
    exit 1
fi

echo ""
echo "Downloading selected files..."
echo "---------------------------------------------"

for i in "${selected_indices[@]}"; do
    file_id=${ids[$i]}
    filename=${names[$i]}

    echo "Downloading $filename..."

    download_url=$(curl -s -D - -o /dev/null \
        "$BASE_URL/api/access/datafile/$file_id" \
        | grep -i '^Location:' | cut -d' ' -f2 | tr -d '\r')

    curl -L -o "$filename" "$download_url"
done

echo "---------------------------------------------"
echo "Download complete."