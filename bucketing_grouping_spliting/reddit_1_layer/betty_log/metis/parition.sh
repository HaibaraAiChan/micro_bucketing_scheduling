#!/bin/bash


# Loop over all files in the folder
for file in *.log; do
    echo "Processing $file..."
    if [ -f "$file" ]; then
        # grep 'partition time' "$file" 
        grep 'partition time' "$file" | awk 'NR > 2 {sum+=$4; count++} END {print sum/count}'
    fi
    echo ""
done






# grep 'range partition time' range_nb_7_hidden_1024_fanout_10,25.log | awk 'NR > 2 {sum+=$4; count++} END {print sum/count}'

# grep 'range partition time' range_nb_8_hidden_1024_fanout_10,25.log | awk 'NR > 2 {sum+=$4; count++} END {print sum/count}'