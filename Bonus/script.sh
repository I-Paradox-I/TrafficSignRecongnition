#!/bin/bash
# Define the hyperparameters
epoch=10
lr=0.0001
weight_decay=0.001
batch_size=64

# Define the model architectures
# models=("lenet" "resnet18" "vgg16" "alexnet" "squeezenet1_0" "vit_b_16" "my_net")
models=("vit_b_16")

# Train and test each model
for model in "${models[@]}"
do
    log_file="Bonus/logs/${model}-${epoch}-${lr}-${batch_size}-${weight_decay}.log"
    echo "Training ${model}..."
    python Bonus/train.py --epoch ${epoch} --lr ${lr} --batch_size ${batch_size}  --weight_decay ${weight_decay} --model ${model} > ${log_file}
done