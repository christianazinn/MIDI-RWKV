#!/bin/bash
#######################################################################################################################
#
# Run demo-training-prepare.sh with the same MODEL_TYPE & N_LAYER & N_EMBD first
# Or, rename your base model to rwkv-init.pth and put it in the output folder
#
# The trainer will load the last rwkv-*.pth in the folder, such that it can continue from a stopped run
# Therefore check the log (### Loading rwkv-xxx.pth... ###), and make sure you don't have extra rwkv-*.pth there
#
#######################################################################################################################
#
MODEL_TYPE="x070" # x070 => rwkv-7.0
#
N_LAYER="12"
N_EMBD="384"
#
CTX_LEN="2048" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="out/"$CTX_LEN"chi-L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
#
#######################################################################################################################
#
M_BSZ="12" # CAN be 16 but for parity w fla it's 8
LR_INIT="1e-4"
LR_FINAL="1e-5"
#
W_DECAY="0.1" # maybe 0.1 actually works better? IDK
BETA_2="0.99"
ADAM_EPS="1e-18"
#
GRAD_CP=0 # 1 => slower, save VRAM; 0 => faster, more VRAM
EPOCH_SAVE=1 # save every 50 "miniepochs" (1 miniepoch = 40320 * ctx_len tokens) => decrease if your GPU is weak
#
#######################################################################################################################
#
N_NODE=1 # number of nodes
GPU_PER_NODE=1 # number of GPUs per node
#
DS_BUCKET_MB=2 # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)
#
python train.py --wandb "" --proj_dir $PROJ_DIR \
 --data_file "/mnt/nvme0n1/pile/pile_20B_tokenizer_text_document" --data_type "binidx" --vocab_size 16000 --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 \
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \
 --num_nodes 1 --micro_bsz 2 --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 --my_exit_tokens 332115325534 --magic_prime 81082817 \
 --lr_init 1e-4 --lr_final 1e-4 --warmup_steps 20 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 \
 --accelerator cpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1

python train.py --load_model $PROJ_DIR"/rwkv-4.pth" --wandb "MIDI-RWKV" --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --data_file "/mnt/nvme0n1/pile/pile_20B_tokenizer_text_document" --my_exit_tokens 332115325534 --magic_prime 81082817 \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 $BETA_2 --adam_eps $ADAM_EPS --my_pile_edecay 0 --data_type "binidx" --vocab_size 16000 \
 --weight_decay $W_DECAY --epoch_save $EPOCH_SAVE --head_size_a 64 \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB
