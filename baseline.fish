 python3 scratch.py 10 ./cifar/cifar100 ./ 100 50000 10000  --image_size 224 --batch_size 16 --epochs 100 --use_augument --model_name xception --n_thread 4 2>&1 | tee xception_scratch.log

 python3 finetuning.py 11 ./cifar/cifar100 ./ 100 50000 10000  --image_size 224 --batch_size 16 --epochs 100 --use_augument --model_name xception --n_thread 4 2>&1 | tee xception_finetuning.log
