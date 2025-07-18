#!/bin/bash

output='./_data/fingerprints.json'
search_dir='./_data/new_source'
pattern='*.mp3'

echo "[" > "$output"
first=1

# Step 1: Get the total number of files
mapfile -d '' files < <(find "$search_dir" -type f -name "$pattern" -print0)
total=${#files[@]}
count=0

# Step 2: Iterate and show progress
for file in "${files[@]}"; do
    # Run fpcalc, add filename, and compact all to one line
    json=$(fpcalc -raw -json "$file" 2>/dev/null | jq --arg fname "$file" '. + {filename: $fname}')
    if [ $first -eq 1 ]; then
        echo "$json" >> "$output"
        first=0
    else
        echo "," >> "$output"
        echo "$json" >> "$output"
    fi

    # Update progress
    count=$((count + 1))
    percent=$((100 * count / total))
    printf "\rProgress: %d/%d files (%d%%)" "$count" "$total" "$percent"
done

echo "]" >> "$output"
echo ""
