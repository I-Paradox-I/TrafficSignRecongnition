REM Define the hyperparameters
set epoch=10
set lr=0.001
set weight_decay=0.001
set batch_size=128

REM Define the model architectures
@REM set models=("lenet" "resnet18" "vgg16" "alexnet" "vit_b_16" "squeezenet1_0" "my_net")
set models=("vit_b_16")

REM Train and test each model
for %%i in %models% do (
    echo Training %%i...
    python Bonus/train.py --epoch %epoch% --lr %lr% --batch_size %batch_size% --weight_decay %weight_decay% --model %%i >> Bonus/logs/%%i-%epoch%-%lr%-%weight_decay%.log
)
