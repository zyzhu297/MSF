export INPUT_DROPOUT_STEP=2000
export NOISY_RA_RATIO=0
export LEARNING_RATE_OF_TASK_PROMPT=0.0005
export NUMBER=25
export BATCH_SIZE=64
export LEARNING_RATE=0.0003
export TOTAL_TRAINING_STEP=5000
export DATA_DIR='datas/'
export DATA_NAME="MVSA_Single"


TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python src/train_more.py \
    --is_few False \
    --few '1' \
    --resume_checkpoint "res/more[1]2_28_lr_1e-5original" \
    --data_dir $DATA_DIR$DATA_NAME \
    --data_name $DATA_NAME \
    --image_grounding_path $DATA_DIR$DATA_NAME"/bing_image_for_"$DATA_NAME".json" \
    --image_input_path $DATA_DIR$DATA_NAME"/blip2_image_feats.lmdb" \
    --text_grounding_path $DATA_DIR$DATA_NAME"/bing_text_for_"$DATA_NAME".json" \
    --text_input_path $DATA_DIR$DATA_NAME"/blip2_text_feats.lmdb" \
    --cold_start_step $INPUT_DROPOUT_STEP \
    --random_p $NOISY_RA_RATIO \
    --tokens_learning_rate $LEARNING_RATE_OF_TASK_PROMPT \
    --output_dir ./res/moredev \
    --use_image True \
    --image_num $NUMBER \
    --use_text True \
    --text_num $NUMBER \
    --do_train False \
    --do_eval False \
    --do_pred True \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 8 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.05 \
    --max_steps $TOTAL_TRAINING_STEP \
    --random_p $NOISY_RA_RATIO \
    --warmup_ratio 0.01 \
    --logging_strategy steps \
    --logging_steps 1000 \
    --logging_first_step False \
    --save_steps 1000 \
    --save_total_limit 1 \
    --seed 0 \
    --dataloader_num_workers 4 \
    --predict_with_generate \
    --generation_max_length 32 \
    --generation_num_beams 5 \
    --overwrite_output_dir True \
    --overwrite_cache True \
    --disable_tqdm False 