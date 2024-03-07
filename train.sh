python fedavg.py \
    -d ucf \
    -m c2fpl\
    --global_testset 0 \
    -bs 128 \
    --global_epoch 50 \
    --test_gap 50 \
    --local_epoch 5 \
    --partition scene_partition_chain_11_V3.pkl \
