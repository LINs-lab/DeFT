"""Test tree KV cache"""

import argparse

# Note(jinwei): test for mem management of dynamic batch size during tree decoding
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lmsys/vicuna-13b-v1.3")
