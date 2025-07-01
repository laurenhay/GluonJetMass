#!/bin/bash

target_dir="mydir"

find "$target_dir" -type f -name '*.txt' | while IFS= read -r file; do
    if [[ "$file" == *"Uncertainty"* ]]; then
        newfile="${file%.txt}.junc.txt"
    else
        newfile="${file%.txt}.jec.txt"
    fi

    echo "Renaming '$file' -> '$newfile'"
    mv -- "$file" "$newfile"
done
