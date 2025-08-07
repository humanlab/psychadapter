# PsychAdapter: Adapting LLM Transformers to Reflect Traits, Personality and Mental Health
This is the source code repository for the paper "PsychAdapter: Adapting LLM Transformers to Reflect Traits, Personality and Mental Health". This work is currently under submission.

This work proposes the architecture PsychAdapter - an transformer-based AI language model that is able to reflect individual characteristics in its text output. PsychAdapter is trained to be able to reflect any of the Big Five personality traits (openness, conscientiousness, extraversion, agreeableness, and neuroticism) as well as mental health variables (depression and life satisfaction), while optionally being conditioned on demographics (e.g., age). The live-demo of our model can be found at: http://3.12.111.1 (not for distributing until manuscript acceptance). 

This project was done in collaboration between PhD students, postdocs, and professors from Stony Brook University (Huy Vu, Swanie Juhng, Adithya Ganesan, Oscar N.E. Kjell, H. Andrew Schwartz), University of Texas at Dallas (Ryan L. Boyd), Stanford University (Johannes C. Eichstaedt), New York University (Joao Sedoc), University of Melbourne (Margaret L. Kern), University of Pennsylvania (Lyle Ungar). Corresponding authors: Huy Vu (hvu@cs.stonybrook.edu), Johannes C. Eichstaedt (johannes.stanford@gmail.com), H. Andrew Schwartz (has@cs.stonybrook.edu).

## Installations requirements
Python: 3.10.0+.

pip install transformers=="4.18.0"

pip install peft=="0.10.0"

## Instructions for training and generating text with PsychAdapter

### Training
We train PsychAdapter using the following command format. The LLM base models can be set through argument `--model_name_or_path`. Run `python3 ./train_psychadapter.py -h` for more information. The code reads the data from `./processed_data` directory then begins the training process. A directory `./trained_models` will be created containing the trained model.
```
# Training Big Five personalities PsychAdapter
python ./codes/train_psychadapter.py \
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
	--num_train_epochs 5
	--save_steps 1000 \
	--logging_steps 100
```

### Inferencing
After training, PsychAdapter can be used to generate text corresponding to all interested dimensions, using the following command. The code loops through all variables and generates text from the high and low value of each variable, controled by the `std_range` and `generate_interval` arguments. There are many configurations for the generating process that can be modifed (e.g., number of generated sentences, nucleous sampling parameters). Run `python3 ./inference_psychadapter.py -h` for more information.
```
# Inferencing Big Five personalities PsychAdapter
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
	--seed 45 \
	--prompting_text "I like to"
```

## How to Cite

If you use this code or model in your research, please cite our paper:

```bibtex
@misc{vu2025psychadapteradaptingllmtransformers,
  title     = {PsychAdapter: Adapting LLM Transformers to Reflect Traits, Personality and Mental Health},
  author    = {Huy Vu and Huy Anh Nguyen and Adithya V Ganesan and Swanie Juhng and Oscar N. E. Kjell and Joao Sedoc and Margaret L. Kern and Ryan L. Boyd and Lyle Ungar and H. Andrew Schwartz and Johannes C. Eichstaedt},
  year      = {2025},
  eprint    = {2412.16882},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url       = {https://arxiv.org/abs/2412.16882}
}

