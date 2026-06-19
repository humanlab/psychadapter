# PsychAdapter: Adapting LLM Transformers to Reflect Traits, Personality and Mental Health
This is the source code repository for the paper "PsychAdapter: Adapting LLM Transformers to Reflect Traits, Personality and Mental Health".

This work proposes the architecture PsychAdapter - an transformer-based AI language model that is able to reflect individual characteristics in its text output. PsychAdapter is trained to be able to reflect any of the Big Five personality traits (openness, conscientiousness, extraversion, agreeableness, and neuroticism) as well as mental health variables (depression and life satisfaction), while optionally being conditioned on demographics (e.g., age).

This project was done in collaboration between PhD students, postdocs, and professors from Stony Brook University (Huy Vu, Huy Anh Nguyen, Swanie Juhng, Adithya Ganesan, Oscar N.E. Kjell, H. Andrew Schwartz), University of Texas at Dallas (Ryan L. Boyd), Stanford University (Johannes C. Eichstaedt), New York University (Joao Sedoc), University of Melbourne (Margaret L. Kern), University of Pennsylvania (Lyle Ungar). Corresponding authors: Huy Vu (hvu@cs.stonybrook.edu), Johannes C. Eichstaedt (johannes.stanford@gmail.com), H. Andrew Schwartz (has@cs.stonybrook.edu).

## HuggingFace Resources
The model checkpoints and the full dataset are available on HuggingFace:
* Model Checkpoints: https://huggingface.co/huvucode/PsychAdapter
* Dataset: https://huggingface.co/datasets/huvucode/PsychAdapter

You can download the dataset and pretrained model checkpoints using the following commands:
```
git clone https://huggingface.co/datasets/huvucode/PsychAdapter data
git clone https://huggingface.co/huvucode/PsychAdapter pretrained_checkpoints
```

Note: large files are stored with Git LFS. Install git-lfs before cloning.

## Installation requirements
```
conda create -n psychadapter python=3.11
conda activate psychadapter
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers peft accelerate pandas tensorboardX
```

### HuggingFace authentication
Some models (including Gemma) require a HuggingFace account and accepted license. Before running:

1. Create an account and generate an access token at https://huggingface.co/settings/tokens
2. Accept the Gemma license at https://huggingface.co/google/gemma-2b
3. Authenticate locally: `huggingface-cli login`

## Instructions for training and generating text with PsychAdapter

### Training
We train PsychAdapter using the following command format. The LLM base models can be set through argument `--model_name_or_path`. Run `python3 ./train_psychadapter.py -h` for more information. The code reads the data from `./processed_data` directory then begins the training process. A directory `./trained_models` will be created containing the trained model.

Note: the first run will process and cache the dataset (~30 mins). Subsequent runs will load from cache and start training immediately.

To obtain training and validating dataset (containing messages' text and their corresponding "estimated" construct scores, e.g. Big Five scores, depression, life-satisfaction scores) for research purpose, please contact Huy Vu at [hvu@cs.stonybrook.edu].


```
# Training Big Five personalities PsychAdapter (single GPU)
python ./src/train_psychadapter.py \
	--train_data_file ./data/big5_training_data.csv \
	--eval_data_file ./data/big5_validating_data.csv \
	--output_dir ./checkpoints/big5_model \
	--model_name_or_path google/gemma-2b \
	--latent_size 5 \
	--do_lower_case \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--gradient_accumulation_steps 2 \
	--do_train \
	--evaluate_during_training \
	--learning_rate 5e-5 \
	--num_train_epochs 5 \
	--save_steps 1000 \
	--logging_steps 100

# Training Big Five personalities PsychAdapter (multi-GPU)
accelerate launch ./src/train_psychadapter.py \
	--train_data_file ./data/big5_training_data.csv \
	--eval_data_file ./data/big5_validating_data.csv \
	--output_dir ./checkpoints/big5_model \
	--model_name_or_path google/gemma-2b \
	--latent_size 5 \
	--do_lower_case \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--gradient_accumulation_steps 2 \
	--do_train \
	--evaluate_during_training \
	--learning_rate 5e-5 \
	--num_train_epochs 5 \
	--save_steps 1000 \
	--logging_steps 100
```

### Inferencing
After training, PsychAdapter can be used to generate text corresponding to all interested dimensions, using the following command. The code loops through all variables and generates text from the high and low value of each variable, controled by the `std_range` and `generate_interval` arguments. There are many configurations for the generating process that can be modifed (e.g., number of generated sentences, nucleous sampling parameters). Run `python3 ./inference_psychadapter.py -h` for more information.
```
# Inferencing Big Five personalities PsychAdapter
python ./src/inference_psychadapter.py \
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
	--seed 45 \
	--prompting_text "I like to"
```

## How to Cite

If you use this code or model in your research, please cite our paper:

```bibtex
@article{vu2026psychadapter,
  title={PsychAdapter: Adapting LLM Transformers to Reflect Traits, Personality and Mental Health},
  author={Vu, Huy and Nguyen, Huy Anh and Ganesan, Adithya V. and Juhng, Swanie and Kjell, Oscar N. E. and Sedoc, Joao and Kern, Margaret L. and Boyd, Ryan L. and Ungar, Lyle and Schwartz, H. Andrew and Eichstaedt, Johannes C.},
  journal={npj Artificial Intelligence},
  volume={2},
  number={7},
  year={2026},
  publisher={Nature Publishing Group},
  doi={10.1038/s44387-026-00071-9},
  url={https://www.nature.com/articles/s44387-026-00071-9}
}
```

