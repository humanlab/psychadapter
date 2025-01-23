# pip install transformers=="4.18.0"
# pip install peft=="0.10.0"

# For variables: big5 personalities
python ./codes/inference_psychadapter.py \
	--train_data_file ./data/big5_training_data.csv \
	--output_dir ./checkpoints/big5_model \
	--model_name_or_path google/gemma-2b \
	--checkpoint_step 30000 \
    --psych_variables big5 \
	--latent_size 5 \
	--do_lower_case \
	--generate_num 10 \
	--generate_length 64 \
	--temperature 0.7 \
	--top_k 10 \
	--top_p 0.9 \
	--std_range 3.0 \
	--generate_interval 3.0 \
	--seed 45
    ** adding prompt
    --prompting_text "I like to"

# For variables: Depression
python ./codes/inference_psychadapter.py \
	--train_data_file ./data/dep_training_data.csv \
	--output_dir ./checkpoints/dep_model \
	--model_name_or_path google/gemma-2b \
	--checkpoint_step 30000 \
    --psych_variables dep \
	--latent_size 1 \
	--do_lower_case \
	--generate_num 10 \
	--generate_length 64 \
	--temperature 0.7 \
	--top_k 10 \
	--top_p 0.9 \
	--std_range 3.0 \
	--generate_interval 3.0 \
	--seed 45

# For variables: Life-satisfaction
python ./codes/inference_psychadapter.py \
	--train_data_file ./data/swl_training_data.csv \
	--output_dir ./checkpoints/swl_model \
	--model_name_or_path google/gemma-2b \
	--checkpoint_step 30000 \
    --psych_variables swl \
	--latent_size 1 \
	--do_lower_case \
	--generate_num 10 \
	--generate_length 64 \
	--temperature 0.7 \
	--top_k 10 \
	--top_p 0.9 \
	--std_range 3.0 \
	--generate_interval 3.0 \
	--seed 45
