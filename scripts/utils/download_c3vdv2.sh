echo "Fetching dataset metadata..."

metadata=$(curl -s -f "$BASE_URL/api/datasets/:persistentId?persistentId=doi:$DOI")

if [ $? -ne 0 ] || [ -z "$metadata" ]; then
    echo "ERROR: Failed to fetch metadata (network / proxy / server issue)"
    exit 1
fi

# Validate JSON before parsing
echo "$metadata" | python3 - <<EOF
import sys, json
try:
    json.load(sys.stdin)
except Exception as e:
    print("ERROR: Invalid JSON received")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "Raw response:"
    echo "$metadata"
    exit 1
fi