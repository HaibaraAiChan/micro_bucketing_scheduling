#!/bin/bash
# Check if the folder exists
# if [ -z "$1" ]; then
#     echo "Usage: $0 ./"
#     exit 1
# fi

# if [ ! -d "$1" ]; then
#     echo "Error: Directory $1 does not exist."
#     exit 1
# fi

# Loop over all files in the folder
for file in *.log; do
    echo "Processing $file..."
    if [ -f "$file" ]; then
        grep 'metis partition spend time' "$file" | awk 'NR > 2 {sum+=$5; count++} END {print sum/count}'
    fi
    echo ""
done






# grep 'range partition time' range_nb_7_hidden_1024_fanout_10,25.log | awk 'NR > 2 {sum+=$4; count++} END {print sum/count}'

# grep 'range partition time' range_nb_8_hidden_1024_fanout_10,25.log | awk 'NR > 2 {sum+=$4; count++} END {print sum/count}'