#!/bin/bash

# Initialize or clear the output file
output_file="mean_data.txt"
> "$output_file"

# Loop through all .log files in the current folder
for file in *bucketing.log; do
  echo "Processing $file..."
  
  collect_time=0
  block_time=0
  sum_array=()
  
  # Read the file line by line
  while IFS= read -r line; do
    
    # Extract "collect connection checking time" value
    if [[ $line == *"connection check time"* ]]; then
      collect_time=$(echo "$line" | awk '{print $NF}')
    fi
    
    # Extract "block generation total time" value
    if [[ $line == *"block generation time"* ]]; then
      block_time=$(echo "$line" | awk '{print $NF}')
      
      # If both values have been captured, sum them
      if [ ! -z "$collect_time" ] && [ ! -z "$block_time" ]; then
        sum=$(echo "$collect_time + $block_time" | bc)
        sum_array+=("$sum")
        
        # Reset the captured times
        collect_time=0
        block_time=0
      fi
    fi
  done < "$file"
echo "sum array: $sum_array "
  # Sort the array and remove min and max
  if [ ${#sum_array[@]} -gt 2 ]; then
    IFS=$'\n' sorted=($(sort -n <<<"${sum_array[*]}"))
    unset IFS
    sorted=("${sorted[@]:1:${#sorted[@]}-2}") # Remove min and max
    
    # Calculate the mean
    sum=0
    for i in "${sorted[@]}"; do
      sum=$(echo "$sum + $i" | bc)
    done
    
    mean=$(echo "scale=4; $sum / ${#sorted[@]}" | bc)
    echo "Mean for $file: $mean" >> "$output_file"
  else
    echo "Not enough data points for $file to remove min and max" >> "$output_file"
  fi
done

echo "Done! Mean values calculated in $output_file."
