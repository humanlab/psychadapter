# pip install transformers=="4.18.0"
# pip install peft=="0.10.0"

# For variables: big5 personalities
python ./codes/train_psychadapter.py \
	--train_data_file ./test_training_data/big5_training_data.csv \
	--eval_data_file ./test_training_data/big5_validating_data.csv \
	--output_dir ./test_training_checkpoints/big5_model \
	--model_name_or_path google/gemma-2b \
	--latent_size 5 \
	--do_lower_case \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--gradient_accumulation_steps 2 \
	--do_train \
	--evaluate_during_training \
	--learning_rate 5e-5 \
	--save_steps 1000 \
	--logging_steps 100 \
	--num_train_epochs 5


# For variables: Depression
python ./codes/train_psychadapter.py \
	--train_data_file ./test_training_data/dep_training_data.csv \
	--eval_data_file ./test_training_data/dep_validating_data.csv \
	--output_dir ./test_training_checkpoints/dep_model \
	--model_name_or_path google/gemma-2b \
	--latent_size 1 \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--gradient_accumulation_steps 2 \
	--do_train \
	--evaluate_during_training \
	--learning_rate 5e-5 \
	--save_steps 1000 \
	--logging_steps 100 \
	--num_train_epochs 5


# For variables: Depression
python ./codes/train_psychadapter.py \
	--train_data_file ./test_training_data/swl_training_data.csv \
	--eval_data_file ./test_training_data/swl_validating_data.csv \
	--output_dir ./test_training_checkpoints/swl_model \
	--model_name_or_path google/gemma-2b \
	--latent_size 1 \
	--do_lower_case \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--gradient_accumulation_steps 2 \
	--do_train \
	--evaluate_during_training \
	--learning_rate 5e-5 \
	--save_steps 1000 \
	--logging_steps 100 \
	--num_train_epochs 5
