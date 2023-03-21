#!/bin/bash


max_seq_length=128
train_batch_size="8"
eval_batch_size="8"
num_train_epochs="50"

export TOKENIZERS_PARALLELISM=true


for j in  'eng'   #'amh' 'eng' 'fra' 'hau' 'ibo' 'lin' 'pcm' 'run' 'swa' 'yor' 'sna' 
do
    for i in  "google/flan-t5-large"  #"castorini/afriteva_base" castorini/afriteva_large"  "masakhane/afri-mt5-base" "masakhane/afri-byt5-base"
    do
      for seed in {0..7}
      do

          train_data_path="../../data/${j}/train-combined.tsv"
          eval_data_path="../../data/${j}/dev-combined.tsv"
          test_data_path="../../data/${j}/test-combined.tsv"

          model_name_or_path=$i
          tokenizer_name_or_path=$i
          output_dir="output_"${i}-${j}
          lang=${j}


          n_gpu=1
          CUDA_VISIBLE_DEVICES=$seed
          learning_rate="3e-4"
          gradient_accumulation_steps="4"
          class_dir=../../data/${j}
          data_column="combined"
          target_column="category"
          prompt="classify: "


          dt=$(date '+%d/%m/%Y %H:%M:%S');
          echo "Start Time: $dt" > /tmp/run-$seed.log

          CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python classification_trainer2.py --train_data_path=$train_data_path \
                  --eval_data_path=$eval_data_path \
                  --test_data_path=$test_data_path \
                  --model_name_or_path=$model_name_or_path \
                  --tokenizer_name_or_path=$tokenizer_name_or_path \
                  --output_dir=$output_dir \
                  --max_seq_length=$max_seq_length \
                  --train_batch_size=$train_batch_size \
                  --eval_batch_size=$eval_batch_size \
                  --num_train_epochs=$num_train_epochs \
                  --gradient_accumulation_steps=$gradient_accumulation_steps \
                  --class_dir=$class_dir \
                  --target_column=$target_column \
                  --data_column=$data_column \
                  --prompt=$prompt \
                  --learning_rate="3e-4" \
                  --weight_decay="0.0" \
                  --adam_epsilon="1e-8" \
                  --warmup_steps="0" \
                  --n_gpu=$n_gpu \
                  --fp_16="false" \
                  --max_grad_norm="1.0" \
                  --opt_level="O1" \
                  --seed=$seed \
                  --lang=$lang &

            sleep 2 &

            dt=$(date '+%d/%m/%Y %H:%M:%S');
            echo "End Time: $dt" > /tmp/run-$seed.log



          # cp -r output_${i}-${j}/ /home/azime/masakhane-news/out/.
          # rm -r output_${i}-${j}/checkpoints/


      done
    done
done

# cp -r ../text2text/output*   /home/azime/.
