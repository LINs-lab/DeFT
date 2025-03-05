#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <device>"
    exit 1
fi


modes=("seq" "flatten" "node" "node_chunk")
mems=("paged")
tasks=("docmergeToT" "sorting128ToT" "set128ToT" "keywordToT")

# prompt_len=("None" "1000" "2000" "4000" "5000")  # for llama3-8B

prompt_len=("None" "1000" "2000" "4000" "6000" "8000" "16000" "20000") # for llama3.1-8B

DEVICES=$1
# Create records directory
mkdir -p ./records
mkdir -p raw_data
# add time function
model="meta-llama/Meta-Llama-3.1-8B"
# model="meta-llama/Meta-Llama-3-8B"

# start timing
start=$(date +%s)
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
record="./records/record_$timestamp.csv"
model_safe=$(echo "$model" | sed 's|/|_|g')
mkdir -p "raw_data/${model_safe}"
# Add header to records file
echo "task,mode,mem,tree_idx,prompt_len,max_seq_len,success" > "$record"

for task in "${tasks[@]}"; do
    for idx in {0..0}; do
        for mode in "${modes[@]}"; do
            for mem in "${mems[@]}"; do
                # Skip invalid combinations
                if [ "$mode" == "tree" ] && [ "$mem" == "paged" ]; then
                    continue
                fi

                for plen in "${prompt_len[@]}"; do
                    # get max_seq_len
                    if [ "$plen" == "None" ]; then
                        max_seq_len="16384"
                    else
                        max_seq_len=$((plen + 2000))
                    fi

                    # Define output file with prompt_len and max_seq_len
                    filename="raw_data/${model_safe}/${task}_${mode}_${mem}_${idx}_p${plen}_s${max_seq_len}.json"

                    # Echo parameters in green
                    echo -e "\033[32mRunning with parameters: --task $task --mode $mode --mem $mem --tree_idx $idx --prompt_len $plen --max_seq_len $max_seq_len --output_file ./$filename\033[0m"

                    # Define the base command
                    CMD="CUDA_VISIBLE_DEVICES=\"$DEVICES\" python ../../examples/run_DeFT_llama_paged.py \
                        --model \"$model\" \
                        --mode \"$mode\" \
                        --max_seq_len \"$max_seq_len\" \
                        --Branch_controller \"Practical_Tree\" \
                        --dataset \"../../../dataset/generation/Reasoning/$task.json\" \
                        --tree_idx \"$idx\" \
                        --output_file \"./$filename\" \
                        --mem \"$mem\" \
                        --port 28851"

                    if [ "$plen" != "None" ]; then
                    CMD="$CMD --prompt_len \"$plen\""
                    fi

                    # execute
                    eval $CMD
                    # Record success
                    success=$?
                    echo "$task,$mode,$mem,$idx,$plen,$max_seq_len,$success" >> "$record"
                done
            done
        done
    done
done

# stop timing
end=$(date +%s)

# print time taken in green
echo -e "\033[32mTime taken: $((end-start)) seconds\033[0m"
