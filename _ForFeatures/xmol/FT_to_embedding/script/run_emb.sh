set -eux

source ./slurm/env.sh
source ./slurm/utils.sh

source ./conf_pre/ft_conf.sh

export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_sync_nccl_allreduce=1

# check
check_iplist

distributed_args="--node_ips ${PADDLE_TRAINERS} \
                --node_id ${PADDLE_TRAINER_ID} \
                --current_node_ip ${POD_IP}"

mkdir -p log

python -u  ./run_classifier.py --use_cuda false \
                   --is_distributed false \
                   --use_fast_executor ${e_executor:-"true"} \
                   --tokenizer ${TOKENIZER:-"FullTokenizer"} \
                   --use_fp16 ${use_fp16:-"false"} \
                   --use_dynamic_loss_scaling ${use_fp16} \
                   --init_loss_scaling ${loss_scaling:-128} \
                   --do_train false \
                   --do_val false \
                   --do_test false \
                   --verbose true \
                   --batch_size 1 \
                   --in_tokens false \
                   --stream_job ${STREAM_JOB:-""} \
                   --init_pretraining_params ${MODEL_PATH:-""} \
                   --init_checkpoint ${CKPT_PATH:-""} \
                   --train_set ${TASK_DATA_PATH}/bindingdb \
                   --test_set ${TASK_DATA_PATH}/bindingdb \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --checkpoints ./checkpoints \
                   --save_steps ${SAVE_STEPS} \
                   --weight_decay 0.01 \
                   --warmup_proportion ${WARMUP_PROPORTION:-"0.0"} \
                   --validation_steps ${VALID_STEPS} \
                   --epoch ${EPOCH} \
                   --max_seq_len 256 \
                   --learning_rate ${LR_RATE:-"1e-4"} \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels ${num_labels} \
                   --random_seed 1 > log/lanch.log 2>&1
