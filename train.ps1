    python .\src\server\fedavg.py `
    -d ucf `
    -m c2fpl_ucf `
    --train_mode US `
    --global_testset 0 `
    -bs 128 `
    --global_epoch 15`
    --local_epoch 1`
    --partition ratio_partition_10_V3.pkl `
    --partition_chain ratio_partition_chain_10_V3.pkl `
    --video_num_partition ratio_video_num_partition_10_V3.pkl `
    --test_gap 1`
    --join_ratio 1`
    --gmm_pl 1`
    --eta_clustering 0`
    --load 0`
    --refine 0`

    # python .\src\server\fedavg.py `
    # -d ucf `
    # -m c2fpl_ucf `
    # --train_mode US `
    # --global_testset 0 `
    # -bs 128 `
    # --global_epoch 10`
    # --local_epoch 1`
    # --partition partition_5_V3.pkl `
    # --partition_chain partition_chain_5_V3.pkl `
    # --video_num_partition video_num_partition_5_V3.pkl `
    # --test_gap 1`
    # --join_ratio 1`
    # --gmm_pl 1`
    # --eta_clustering 0`
    # --load 1`
    # --refine 0`

    # python .\src\server\fedavg.py `
    # -d ucf `
    # -m c2fpl `
    # --train_mode US `
    # --global_testset 0 `
    # -bs 128 `
    # --global_epoch 10`
    # --local_epoch 1`
    # --partition scene_partition_11_V3.pkl ` 
    # --partition_chain scene_partition_chain_11_V3.pkl ` 
    # --video_num_partition scene_video_num_partition_11_V3.pkl `
    # --test_gap 1`
    # --join_ratio 1`
    # --gmm_pl 1`
    # --eta_clustering 0`
    # --load 0`
    # --refine 1`

