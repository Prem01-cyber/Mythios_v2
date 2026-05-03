#!/bin/bash

set -e

SPLIT_DIR="vuln_data_split"
VULN_DIR="vuln_data"

echo "Creating vuln_data directory..."
mkdir -p "$VULN_DIR"

echo ""
echo "Reassembling dataset files..."
echo "============================="

find "$SPLIT_DIR" -type f | while read -r file; do
    rel_path="${file#$SPLIT_DIR/}"
    dir_path=$(dirname "$rel_path")
    filename=$(basename "$file")
    
    target_dir="$VULN_DIR/$dir_path"
    mkdir -p "$target_dir"
    
    if [[ $filename == *.part ]]; then
        base_name="${filename%.*.*}"
        
        if [ ! -f "$target_dir/$base_name" ]; then
            first_part="$SPLIT_DIR/$dir_path/${base_name}.00.part"
            if [ -f "$first_part" ] && [[ $filename == *.00.part ]]; then
                echo ""
                echo "Reassembling: $dir_path/$base_name"
                cat "$SPLIT_DIR/$dir_path/${base_name}".*.part > "$target_dir/$base_name"
                size=$(stat -c%s "$target_dir/$base_name" | numfmt --to=iec-i --suffix=B 2>/dev/null || echo "unknown")
                echo "  Complete: $size"
            fi
        fi
    else
        if [ ! -f "$target_dir/$filename" ]; then
            echo ""
            echo "Copying: $dir_path/$filename"
            cp "$file" "$target_dir/$filename"
        fi
    fi
done

echo ""
echo "============================="
echo "Reassembly complete!"
echo "Files saved to: $VULN_DIR"
echo ""
echo "Directory structure:"
tree -h "$VULN_DIR" 2>/dev/null || find "$VULN_DIR" -type f -exec ls -lh {} \;
