# stepwisepathnet_tpu
TPU implementation


## よく使うコマンド
```
python3 sw-pathnet-mod_tournament_tpu.py 0 ./cifar/cifar10 ./ 10 50000 10000  --image_size 224 --batch_size 128 --tpu_name $TPU_NAME --epochs 2 2>&1 | tee test.log
```

```
 python3 sw-pathnet-mod_tournament.py 0 ./cifar/cifar100 ./ 100 50000 10000  --image_size 224 --batch_size 64 --epochs 100 --use_augument --model_name inceptionv3 --n_thread 8 2>&1 | tee inceptionv3.log
 ```
