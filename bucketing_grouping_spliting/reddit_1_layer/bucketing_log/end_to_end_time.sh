#!/bin/bash

# Initialize or clear the output file
output_file="end_to_end_mean_data.txt"
> "$output_file"

# Loop through all .log files in the current folder
for file in *bucketing.log; do
  echo "Processing $file..."

  sum_array=()

  # Read the file line by line
  while IFS= read -r line; do

    # Extract "collect connection checking time" value
    if [[ $line == *"end to end time"* ]]; then
      end_time=$(echo "$line" | awk '{print $NF}')
      sum_array+=("$end_time")
    fi

  done < "$file"

  # Calculate the sum and number of elements
  sum=0
  num_elements=${#sum_array[@]}
  max=${sum_array[0]}
  min=${sum_array[0]}

  for value in "${sum_array[@]}"; do
    sum=$(bc <<< "$sum + $value")
    if (( $(bc <<< "$value > $max") )); then
      max="$value"
    fi
    if (( $(bc <<< "$value < $min") )); then
      min="$value"
    fi
  done

  # Remove max and min values
  sum=$(bc <<< "$sum - $max - $min")

  # Calculate the mean
  mean=$(bc <<< "scale=2; $sum / ($num_elements - 2)")

  # Append the mean to the output file
  echo "Mean for $file:$mean" >> "$output_file"

done
