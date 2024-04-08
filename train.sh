# python fedavg.py \
#     -d ucf \
#     -m c2fpl\
#     --global_testset 0 \
#     -bs 128 \
#     --global_epoch 50 \
#     --test_gap 50 \
#     --local_epoch 5 \
#     --partition scene_partition_chain_11_V3.pkl \

python fedavg.py \
    -d ucf \
    -m c2fpl_ucf \
    --train_mode US \
    --global_testset 0 \
    -bs 128 \
    --global_epoch 15\
    --local_epoch 1\
    --partition ratio_partition_10_V3.pkl \
    --partition_chain ratio_partition_chain_10_V3.pkl \
    --video_num_partition ratio_video_num_partition_10_V3.pkl \
    --test_gap 1\
    --join_ratio 1\
    --gmm_pl 1\
    --eta_clustering 0\
    --load 0\
    --refine 0\