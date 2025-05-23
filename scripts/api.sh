export INPUT_DROPOUT_STEP=0
export NOISY_RA_RATIO=0.3
export LEARNING_RATE_OF_TASK_PROMPT=0.0005
export NUMBER=8
export BATCH_SIZE=64
export LEARNING_RATE=0.00001
export TOTAL_TRAINING_STEP=300
export DATA_DIR='datas/'
export DATA_NAME="MVSA_Single"
export STEP=10
export OUT_DIR="5_20"
export FEW='1'

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python src/Semi.py \
    --is_few True \
    --few $FEW \
    --data_dir $DATA_DIR$DATA_NAME \
    --data_name $DATA_NAME  \
    --resume_checkpoint "res/more[1]4_20.1original" \
    --image_grounding_path $DATA_DIR$DATA_NAME"/bing_image_for_"$DATA_NAME".json" \
    --image_input_path $DATA_DIR$DATA_NAME"/blip2_image_feats.lmdb" \
    --text_grounding_path $DATA_DIR$DATA_NAME"/bing_text_for_"$DATA_NAME".json" \
    --text_input_path $DATA_DIR$DATA_NAME"/blip2_text_feats.lmdb" \
    --cold_start_step $INPUT_DROPOUT_STEP \
    --random_p $NOISY_RA_RATIO \
    --tokens_learning_rate $LEARNING_RATE_OF_TASK_PROMPT \
    --output_dir ./res/more_logits \
    --use_image True \
    --image_num $NUMBER \
    --use_text True \
    --text_num $NUMBER \
    --do_train False \
    --do_eval True \
    --do_pred True \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 8 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.05 \
    --max_steps $TOTAL_TRAINING_STEP \
    --random_p $NOISY_RA_RATIO \
    --warmup_ratio 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_first_step False \
    --save_steps 10 \
    --save_total_limit 1 \
    --seed 0 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --dataloader_num_workers 4 \
    --load_best_model_at_end True \
    --metric_for_best_model average \
    --predict_with_generate \
    --generation_max_length 32 \
    --generation_num_beams 5 \
    --overwrite_output_dir True \
    --overwrite_cache True \
    --disable_tqdm False 