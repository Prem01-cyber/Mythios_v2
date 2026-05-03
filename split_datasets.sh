#!/bin/bash

set -e

VULN_DIR="vuln_data"
SPLIT_DIR="vuln_data_split"

echo "Creating split directory..."
mkdir -p "$SPLIT_DIR"

echo ""
echo "Splitting large dataset files..."
echo "================================"

find "$VULN_DIR" -type f \( -name "*.csv" -o -name "*.json" \) | while read -r file; do
    rel_path="${file#$VULN_DIR/}"
    
    if [[ "$rel_path" == raw/* ]]; then
        echo "Skipping raw file: $rel_path"
        continue
    fi
    
    filesize=$(stat -c%s "$file")
    filesize_mb=$((filesize / 1024 / 1024))
    
    echo ""
    echo "Processing: $rel_path (${filesize_mb}MB)"
    
    target_dir="$SPLIT_DIR/$(dirname "$rel_path")"
    mkdir -p "$target_dir"
    
    filename=$(basename "$file")
    
    if [ $filesize_mb -gt 50 ]; then
        echo "  Splitting into 50MB chunks..."
        split -b 50M -d "$file" "$target_dir/${filename}." --additional-suffix=".part"
        parts_count=$(ls -1 "$target_dir/${filename}".*.part 2>/dev/null | wc -l)
        echo "  Created $parts_count parts"
    else
        echo "  File is small enough, copying as-is..."
        cp "$file" "$target_dir/$filename"
    fi
done

echo ""
echo "================================"
echo "Split complete!"
echo "Files saved to: $SPLIT_DIR"
echo ""
echo "Directory structure:"
tree -h "$SPLIT_DIR" 2>/dev/null || find "$SPLIT_DIR" -type f -exec ls -lh {} \;
