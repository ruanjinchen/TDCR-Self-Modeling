'''
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------SIM 2------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

sim 2节 有base的，5000组数据 mlp版本 有RGB H100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
评估结果
[Demo] Done. Samples=500 mean_CD=0.00009624 std_CD=0.00000414 mean_EMD=0.00236875 std_EMD=0.00101577
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
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
评估结果
[Demo] Done. Samples=500 mean_CD=0.00003816 std_CD=0.00001183 mean_EMD=0.00063591 std_EMD=0.00030735
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
export CUDA_VISIBLE_DEVICES=1
python train_flowmatching.py \
  --data_dir datasets/sim/2m_with_base \
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
  --out_dir final_results/sim_2m_with_base_hybrid_1_2


sim 2节 没有base的，5000组数据 hybrid版本 有RGB A100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0480] CD-L2(cond) mean = 0.000020
export CUDA_VISIBLE_DEVICES=1
python train_flowmatching.py \
  --data_dir datasets/sim/2m_no_base \
  --batch_size 16 --lr 6e-4 --warmup_steps 4000 --epochs 500 --save_every 20 \
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
  --out_dir runs_final/sim_2m_no_base_hybrid_1_4

-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------SIM 3------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
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
更新了最新的Hybrid参数之后重新跑
export CUDA_VISIBLE_DEVICES=1
python train_flowmatching.py \
  --data_dir datasets/sim/3m_with_base \
  --batch_size 16 --lr 6e-4 --warmup_steps 4000 --epochs 500 --save_every 20 \
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
  --out_dir runs_final/sim_3m_with_base_hybrid_1_3

  
sim 3节 没有base的，5000组数据 hybrid版本 有RGB H100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
[epoch 0500] CD-L2(cond) mean = 0.000024
export CUDA_VISIBLE_DEVICES=4
python train_flowmatching.py \
  --data_dir datasets/sim/3m_no_base \
  --batch_size 16 --lr 6e-4 --warmup_steps 4000 --epochs 500 --save_every 20 \
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
  --out_dir runs_final/sim_3m_no_base_hybrid_1_4
  

-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------SIM 5------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

sim 5节 有base的，10000组数据 mlp版本 有RGB H100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
评估完成 
[Demo] Done. Samples=1000 mean_CD=0.00006432 std_CD=0.00000758 mean_EMD=0.00198901 std_EMD=0.00111620
[Demo] Done. Samples=1000 mean_CD=0.00008781 std_CD=0.00000708 mean_EMD=0.00200824 std_EMD=0.00111626
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

sim5节 有base的，10000组数据，Hybrid版本，全量参数，H100，训练完成
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
评估完成 
[Demo] Done. Samples=1000 mean_CD=0.00006055 std_CD=0.00001052 mean_EMD=0.00191688 std_EMD=0.00110042
[Demo] Done. Samples=1000 mean_CD=0.00008157 std_CD=0.00000960 mean_EMD=0.00193422 std_EMD=0.00110057
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
  --out_dir runs_final/sim_5m_with_base_hybrid_1_2

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

  
sim 5节 没有base的，10000组数据 hybrid版本 有RGB H100：
最新的Hybrid的参数 重新跑
export CUDA_VISIBLE_DEVICES=1
python train_flowmatching.py \
  --data_dir datasets/sim/5m_no_base \
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
  --out_dir runs_final/sim_5m_no_base_hybrid_1_3



-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------REAL 2-----------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

real 2节 有base的，10000组数据 mlp版本 有RGB A100：
√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√
评估结果
[Demo] Done. Samples=1000 mean_CD=0.00020236 std_CD=0.00003135 mean_EMD=0.00327785 std_EMD=0.00122790
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
重跑中，最新的Hybrid的参数, 训练进行中
export CUDA_VISIBLE_DEVICES=5
python train_flowmatching.py \
  --data_dir datasets/real/2m_with_base \
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
  --out_dir runs_final/real_2m_with_base_hybrid_1_2

-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------REAL 3-----------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
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
重跑中，更新了Hybrid的参数
export CUDA_VISIBLE_DEVICES=4
python train_flowmatching.py \
  --data_dir datasets/real/3m_with_base \
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
  --out_dir runs_final/real_3m_with_base_hybrid_1_2

'''