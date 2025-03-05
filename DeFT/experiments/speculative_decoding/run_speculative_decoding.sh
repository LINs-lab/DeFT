#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <device>"
    exit 1
fi

# Define parameters
# for unpaged case
# mems=("unpaged")
# modes=("tree" "seq")

# for paged case
mems=("paged")
modes=("seq" "flatten" "node" "node_chunk" "tree_index")


sizes=("32" "64" "128" "256")
prompt_len=("None" "1000" "2000" "4000" "6000" "8000" "16000" "20000") #for llama3.1-8B
task="speculative_decoding"
DEVICES=$1
# model="meta-llama/Meta-Llama-3-8B"
model="meta-llama/Meta-Llama-3.1-8B"
model_safe=$(echo "$model" | sed 's|/|_|g')

# Create required directories
mkdir -p ./records
mkdir -p raw_data
mkdir -p "raw_data/${model_safe}"
# Start timing
start=$(date +%s)
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
record="./records/record_$timestamp.csv"

# Add header to records file
echo "task,mode,mem,size,success" > "$record"

for plen in "${prompt_len[@]}"; do
    for size in "${sizes[@]}"; do
        for mode in "${modes[@]}"; do
            for mem in "${mems[@]}"; do
                # Skip invalid combinations
                if [ "$mode" == "tree" ] && [ "$mem" == "paged" ]; then
                    continue
                fi
                if [ "$mode" != "tree" ] && [ "$mem" == "unpaged" ]; then
                    continue
                fi

                if [ "$plen" == "None" ]; then
                    max_seq="6000"
                else
                    max_seq=$((plen + 1000))
                fi
                # Define output file
                filename="raw_data/${model_safe}/${task}_${mode}_${mem}_${size}_p${plen}_s${max_seq}.json"

                # Echo parameters in green
                echo -e "\033[32mRunning with parameters: --task $task --mode $mode --mem $mem tree size $size --output_file $filename\033[0m"

                CMD="CUDA_VISIBLE_DEVICES=\"$DEVICES\" python ../../examples/run_DeFT_llama_paged.py \
                --model \"$model\" \
                --mode \"$mode\" \
                --max_seq_len \"$max_seq\" \
                --Branch_controller \"Speculative_Decoding\" \
                --dataset \"../../../dataset/generation/Speculative_Decoding/APPS_tree_size${size}.json\" \
                --output_file \"$filename\" \
                --mem \"$mem\" \
                --port 28871"


                if [ "$plen" != "None" ]; then
                    CMD="$CMD --prompt_len \"$plen\""
                fi

                # execute
                eval $CMD

                # Record success
                success=$?
                echo "$task,$mode,$mem,$size,$success" >> "$record"
            done
        done
    done
done
# Stop timing
end=$(date +%s)

# Print elapsed time in green
echo -e "\033[32mTime taken: $((end-start)) seconds\033[0m"
