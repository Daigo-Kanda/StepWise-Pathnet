for i in (seq 0 4)
    python3 sw-pathnet-mod_tournament.py $i ./cifar/cifar100 ./ 100 50000 10000  --image_size 224 --batch_size 16 --epochs 30 --use_augument --model_name xception --n_thread 8 2>&1 | tee xception_mod-tournamet_$i.log
end
