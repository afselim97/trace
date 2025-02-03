#!/bin/bash
# L=20000

pops=""
n=""
output=""
max_time=""

# Parse command-line arguments
while getopts "p:n:o:t:" opt; do
  case $opt in
    p) pops=$OPTARG ;;
    n) n=$OPTARG ;;
    o) output=$OPTARG ;;
    t) max_time=$OPTARG ;;
    \?) echo "Usage: $0 -p pops -n n -o output -t max_time" >&2
        exit 1 ;;
  esac
done

# Check if all required arguments are provided
if [ -z "$pops" ] || [ -z "$n" ] || [ -z "$output" ] || [ -z "$max_time" ]; then
  echo "Usage: $0 -p pops -n n -o output -m max_time"
  exit 1
fi

python simplify_trees.py -input trees/human/wohns -output trees/human/"$output" -populations $pops -individuals_per_pop $n
../src/run_trasp -input trees/human/"$output" -output results/"$output" -metadata trees/human/"$output"/metadata.csv -mode inferred -min_time 1 -max_time $max_time -num_timepoints 200 -delta 100 -log_time