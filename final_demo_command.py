'''

待执行 记得剪切再粘贴执行
export CUDA_VISIBLE_DEVICES=0
python demo_generate_tdcr.py \
  --ckpt runs_final/sim_5m_no_base_hybrid_1_3/ckpts/latest.pt \
  --data_dir datasets/sim/5m_no_base \
  --split test \
  --demo_out TEST/sim_5m_no_base_hybrid_1_3 \
  --cd_points 4096 \
  --use_emd \
  --emd_points 4096 \
  --sample_steps 100 \
  --batch_size 16

'''


