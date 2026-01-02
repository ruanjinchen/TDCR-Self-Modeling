'''
export CUDA_VISIBLE_DEVICES=1
python demo_generate_tdcr.py \
  --ckpt final_results/sim_5m_with_base_hybrid_1_2/ckpts/latest.pt \
  --data_dir datasets/sim/5m_with_base \
  --split test \
  --demo_out final_demo/sim_5m_with_base_hybrid_1_2 \
  --cd_points 4096 \
  --use_emd \
  --emd_points 4096 \
  --sample_steps 100 \
  --batch_size 64

export CUDA_VISIBLE_DEVICES=1
python demo_generate_tdcr.py \
  --ckpt final_results/sim_5m_with_base_mlp_12_27/ckpts/latest.pt \
  --data_dir datasets/sim/5m_with_base \
  --split test \
  --demo_out final_demo/sim_5m_with_base_mlp_12_27 \
  --cd_points 4096 \
  --use_emd \
  --emd_points 4096 \
  --sample_steps 100 \
  --batch_size 64

export CUDA_VISIBLE_DEVICES=0
python demo_generate_tdcr.py \
  --ckpt final_results/real_2m_with_base_mlp_12_29/ckpts/latest.pt \
  --data_dir datasets/sim/5m_with_base \
  --split test \
  --demo_out final_demo/real_2m_with_base_mlp_12_29 \
  --cd_points 4096 \
  --use_emd \
  --emd_points 4096 \
  --sample_steps 100 \
  --batch_size 64

'''


