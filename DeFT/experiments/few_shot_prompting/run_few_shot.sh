#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <device>"
    exit 1
fi

# Define parameters
modes=("seq" "flatten" "node" "node_chunk")

mems=("paged")
widths=("20" "30" "50")
task="few_shot"
DEVICES=$1


# Create records directory
mkdir -p ./records
mkdir -p raw_data
# Start timing
start=$(date +%s)
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
record="./records/record_$timestamp.csv"
# model="meta-llama/Meta-Llama-3-8B"
# prompt_len=("4000" "5000" "6000" "7000") # for llama3-8B
model="meta-llama/Meta-Llama-3.1-8B"
prompt_len=("1000" "2000" "4000" "6000" "8000" "16000" "20000") # in the paper, the prompt len=4000.
model_safe=$(echo "$model" | sed 's|/|_|g')
mkdir -p "raw_data/${model_safe}"
# export CUDA_VISIBLE_DEVICES=$DEVICES
# export TORCH_CUDA_ARCH_LIST="8.9"

# Add success records
echo "task,mode,mem,width,success" > "$record"

for width in "${widths[@]}"; do
    for mode in "${modes[@]}"; do
        for mem in "${mems[@]}"; do
            # Skip invalid combinations
            if [ "$mode" == "tree" ] && [ "$mem" == "paged" ]; then
                continue
            fi
            if [ "$mode" != "tree" ] && [ "$mem" == "unpaged" ]; then
                continue
            fi

            for plen in "${prompt_len[@]}"; do
                # get max_seq_len
                if [ "$plen" == "None" ]; then
                    max_seq_len="4400"
                else
                    max_seq_len=$((plen + 400))
                fi

                # Define output file with prompt_len and max_seq_len
                filename="raw_data/${model_safe}/${task}_${mode}_${mem}_${width}_p${plen}_s${max_seq_len}.json"

                # Echo in green
                echo -e "\033[32mRunning with parameters: --task $task --mode $mode --mem $mem --width $width --prompt_len $plen --max_seq_len $max_seq_len --output_file $filename\033[0m"

                # Define the base command
                CMD="CUDA_VISIBLE_DEVICES=\"$DEVICES\" python ../../examples/run_DeFT_llama_paged.py \
                    --model \"$model\" \
                    --mode \"$mode\" \
                    --max_seq_len \"$max_seq_len\" \
                    --Branch_controller \"Simple_Tree\" \
                    --max_width \"$width\" \
                    --output_file \"$filename\" \
                    --mem \"$mem\" \
                    --port 28860"

                if [ "$plen" != "None" ]; then
                    CMD="$CMD --prompt_len \"$plen\""
                fi

                # execute
                eval $CMD
                # Record success for Python
                success=$?
                echo "$task,$mode,$mem,$width,$plen,$max_seq_len,$success" >> "$record"
            done
        done
    done
done

# Stop timing
end=$(date +%s)

# Print time taken in green
echo -e "\033[32mTime taken: $((end-start)) seconds\033[0m"
