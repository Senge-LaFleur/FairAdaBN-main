# Baseline models
CUDA_VISIBLE_DEVICES=0 python main_baseline.py --exp_id 3 --rand_seed 0 --max_epoch 400
CUDA_VISIBLE_DEVICES=0 python main_baseline.py --exp_id 4 --rand_seed 0 --max_epoch 400
CUDA_VISIBLE_DEVICES=0 python main_baseline.py --exp_id 5 --rand_seed 0 --max_epoch 400

# FairAdaBN models
CUDA_VISIBLE_DEVICES=0 python main.py --exp_id 3 --rand_seed 0 --max_epoch 400
CUDA_VISIBLE_DEVICES=0 python main.py --exp_id 4 --rand_seed 0 --max_epoch 400
CUDA_VISIBLE_DEVICES=0 python main.py --exp_id 5 --rand_seed 0 --max_epoch 400

# FairAdaBN + Lmi models
CUDA_VISIBLE_DEVICES=0 python main_Lmi.py --exp_id 3 --rand_seed 0 --max_epoch 400
CUDA_VISIBLE_DEVICES=0 python main_Lmi.py --exp_id 4 --rand_seed 0 --max_epoch 400
CUDA_VISIBLE_DEVICES=0 python main_Lmi.py --exp_id 5 --rand_seed 0 --max_epoch 400

# Note: Use a different exp_id for each experiment to avoid overwriting
# For example, for experiment 3, use:
# CUDA_VISIBLE_DEVICES=0 python main_baseline.py --exp_id 3 --rand_seed 0 --max_epoch 400
# for experiment 4, use:
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_id 4 --rand_seed 0 --max_epoch 400
# for experiment 5, use:
# CUDA_VISIBLE_DEVICES=0 python main_Lmi.py --exp_id 5 --rand_seed 0 --max_epoch 400

# Evaluation
python evaluation.py --exp_id 3 --rand_seed 0 --method best # best model on validation set
python evaluation.py --exp_id 3 --rand_seed 0 --method final # Last epoch checkpoint
python evaluation.py --exp_id 4 --rand_seed 0 --method best # best model on validation set
python evaluation.py --exp_id 4 --rand_seed 0 --method final # Last epoch checkpoint
python evaluation.py --exp_id 5 --rand_seed 0 --method best # best model on validation set
python evaluation.py --exp_id 5 --rand_seed 0 --method final # Last epoch checkpoint

# FATE: Baseline vs FairAdaBN
python fate_eval.py --baseline_exp 3 --mitigation_exp 4 --rand_seed 0 --method best
python fate_eval.py --baseline_exp 3 --mitigation_exp 4 --rand_seed 0 --method final

# FATE: Baseline vs FairAdaBN+Lmi
python fate_eval.py --baseline_exp 3 --mitigation_exp 5 --rand_seed 0 --method best
python fate_eval.py --baseline_exp 3 --mitigation_exp 5 --rand_seed 0 --method final

!python /kaggle/working/FairAdaBN-main/fate_eval.py --baseline_exp 3 --mitigation_exp 5 --rand_seed 0 --method best
