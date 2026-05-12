#!/bin/bash


export CUDA_VISIBLE_DEVICES=5


eps_values=(3)

for eps in "${eps_values[@]}"
do
  nohup python3 main.py \
      --dataset strategyqa \
      --eps $eps \
      --top_k 20 \
      --embedding_type glove.6B.300d \
      --save_stop_words True > log_eps${eps}.log 2>&1 &
done