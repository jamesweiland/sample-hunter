#!/bin/bash

# Find all files, compute their md5 hash and path, then process to group duplicates
find ./_data/source -type f -exec md5sum {} + \
| awk '{ hash=$1; file=$2; hashes[hash]=hash in hashes ? hashes[hash] RS file : file; count[hash]++ }
       END { for (h in count) if (count[h]>1) 
         printf "Duplicate Files (MD5:%s):\n%s\n\n", h, hashes[h] }'
