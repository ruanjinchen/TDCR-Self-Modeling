'''
sim 2节 带有base的，5000组数据 mlp版本：

python train_flowmatching.py \
  --data_dir datasets/sim_2m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --emb_dim 64 --width 256 --depth 4 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --out_dir runs_final/sim_2m_with_base_mlp_12_26


sim 3节 带有base的，5000组数据 mlp版本：

python train_flowmatching.py \
  --data_dir datasets/sim_3m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --emb_dim 64 --width 256 --depth 4 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --out_dir runs_final/sim_3m_with_base_mlp_12_26

'''