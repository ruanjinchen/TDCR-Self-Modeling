'''

sim 2节 有base的，5000组数据 mlp版本 有RGB H100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0500] CD-L2(cond) mean = 0.000034
export CUDA_VISIBLE_DEVICES=5
python train_flowmatching.py \
  --data_dir datasets/sim/2m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_2m_with_base_mlp_12_28

sim 2节 没有base的，5000组数据 mlp版本 有RGB A100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0500] CD-L2(cond) mean = 0.000023
export CUDA_VISIBLE_DEVICES=3
python train_flowmatching.py \
  --data_dir datasets/sim/2m_no_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_2m_no_base_mlp_12_28


sim 2节 有base的，5000组数据 hybrid版本 有RGB H100：
ing
export CUDA_VISIBLE_DEVICES=5
python train_flowmatching.py \
  --data_dir datasets/sim/2m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 64 --width 256 --depth 4 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_2m_with_base_hybrid_12_28


sim 2节 没有base的，5000组数据 hybrid版本 有RGB A100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0480] CD-L2(cond) mean = 0.000020
export CUDA_VISIBLE_DEVICES=3
python train_flowmatching.py \
  --data_dir datasets/sim/2m_no_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 64 --width 256 --depth 4 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_2m_no_base_hybrid_12_29

-------------------------------------------------------------------------------------------------------------------------



sim 3节 有base的，5000组数据 mlp版本 有RGB H100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0500] CD-L2(cond) mean = 0.000029
export CUDA_VISIBLE_DEVICES=1
python train_flowmatching.py \
  --data_dir datasets/sim/3m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_3m_with_base_mlp_12_27


sim 3节 没有base的，5000组数据 mlp版本 有RGB H100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0500] CD-L2(cond) mean = 0.000025
export CUDA_VISIBLE_DEVICES=5
python train_flowmatching.py \
  --data_dir datasets/sim/3m_no_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_3m_no_base_mlp_12_27

  
sim 3节 有base的，5000组数据 hybrid版本 有RGB H100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0500] CD-L2(cond) mean = 0.000025
export CUDA_VISIBLE_DEVICES=1
python train_flowmatching.py \
  --data_dir datasets/sim/3m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 64 --width 256 --depth 4 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_3m_with_base_hybrid_12_28

  
sim 3节 没有base的，5000组数据 hybrid版本 有RGB H100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0500] CD-L2(cond) mean = 0.000024
export CUDA_VISIBLE_DEVICES=1
python train_flowmatching.py \
  --data_dir datasets/sim/3m_no_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 64 --width 256 --depth 4 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_3m_no_base_hybrid_12_28
  

-------------------------------------------------------------------------------------------------------------------------



sim 5节 有base的，10000组数据 mlp版本 有RGB H100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0500] CD-L2(cond) mean = 0.000029
export CUDA_VISIBLE_DEVICES=4
python train_flowmatching.py \
  --data_dir datasets/sim/5m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_5m_with_base_mlp_12_27

sim 5节 没有base的，10000组数据 mlp版本 有RGB A100：
[epoch 0500] CD-L2(cond) mean = 0.000028
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
export CUDA_VISIBLE_DEVICES=3
python train_flowmatching.py \
  --data_dir datasets/sim/5m_no_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_5m_no_base_mlp_12_27


sim 5节 有base的，10000组数据 hybrid版本 有RGB H100：
停止于Epoch 363 12月30日上午9点40分
[epoch 0340] CD-L2(cond) mean = 0.000150
[epoch 0360] CD-L2(cond) mean = 0.000203
下面的参数得到的最低的：[epoch 0300] CD-L2(cond) mean = 0.000101
参数不太行
export CUDA_VISIBLE_DEVICES=4
python train_flowmatching.py \
  --data_dir datasets/sim/5m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 64 --width 256 --depth 4 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_5m_with_base_hybrid_12_28

export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 --master_port=29511 \
  train_flowmatching.py \
  --data_dir datasets/sim/5m_with_base \
  --batch_size 4 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 64 --width 256 --depth 4 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 48 32 16 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 --train_fraction 0.625 \
  --out_dir runs_final/sim_5m_with_base_hybrid_12_30_new_hybrid_params_half_data

export CUDA_VISIBLE_DEVICES=2,3
torchrun --standalone --nproc_per_node=2 --master_port=29511 \
  train_flowmatching.py \
  --data_dir sim/5m_with_base \
  --batch_size 4 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 64 --width 256 --depth 4 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 48 32 16 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_5m_with_base_hybrid_12_30_new_hybrid_params_new_data

export CUDA_VISIBLE_DEVICES=4,5
torchrun --standalone --nproc_per_node=2 --master_port=29511 \
  train_flowmatching.py \
  --data_dir datasets/sim/5m_with_base \
  --batch_size 4 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 256 --width 512 --depth 6 --cfg_drop_p 0.0 \
  --ctx_dim 64 \
  --ctx_emb_dim 256 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.80 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_5m_with_base_hybrid_bighead_tau0.8

export CUDA_VISIBLE_DEVICES=1
python train_flowmatching.py \
  --data_dir datasets/sim/5m_with_base \
  --batch_size 32 --lr 8e-4 --warmup_steps 4000 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 256 --width 512 --depth 6 --cfg_drop_p 0.0 \
  --ctx_dim 64 \
  --ctx_emb_dim 256 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 50 \
  --out_dir runs_final/sim_5m_with_base_hybrid_bighead_tau0.97

  
sim 5节 没有base的，10000组数据 hybrid版本 有RGB H100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0500] CD-L2(cond) mean = 0.000055
export CUDA_VISIBLE_DEVICES=1
python train_flowmatching.py \
  --data_dir datasets/sim/5m_no_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 64 --width 256 --depth 4 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/sim_5m_no_base_hybrid_12_29

-------------------------------------------------------------------------------------------------------------------------


real 2节 有base的，10000组数据 mlp版本 有RGB A100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0460] CD-L2(cond) mean = 0.000109
export CUDA_VISIBLE_DEVICES=2
python train_flowmatching.py \
  --data_dir datasets/real/2m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/real_2m_with_base_mlp_12_29

real 2节 有base的，10000组数据 hybrid版本 有RGB H100：
新参数，匹配了MLP的width depth emb_dim, 训练进行中
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 --master_port=29511 \
  train_flowmatching.py \
  --data_dir datasets/real/2m_with_base \
  --batch_size 4 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 256 --width 512 --depth 6 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/real_2m_with_base_hybrid_1_1

-------------------------------------------------------------------------------------------------------------------------


real 3节 有base的，10000组数据 mlp版本 有RGB H100：
[epoch 0500] CD-L2(cond) mean = 0.000088
export CUDA_VISIBLE_DEVICES=4
python train_flowmatching.py \
  --data_dir datasets/real/3m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/real_3m_with_base_mlp_12_30

real 2节 有base的，10000组数据 hybrid版本 有RGB H100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
epoch 0500] CD-L2(cond) mean = 0.000098
export CUDA_VISIBLE_DEVICES=1
python train_flowmatching.py \
  --data_dir datasets/real/3m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 64 --width 256 --depth 4 --cfg_drop_p 0.0 \
  --ctx_dim 16 \
  --ctx_emb_dim 64 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs_final/real_3m_with_base_hybrid_12_30
'''