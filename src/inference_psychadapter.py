# import libraries
import argparse
import logging
import os
import random
import numpy as np
import torch
import pandas as pd

from psychadapter import PsychAdapter
from peft import PeftModel

logger = logging.getLogger(__name__)


"""====================== METHODS DEFINITIONS ======================"""

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def run_inference(model, args, device):
    
    # retrieve embeddings
    data_df = pd.read_csv(args.train_data_file, header = 0, index_col = 0)
    training_sentences_embeddings = data_df.iloc[:,1:].values

    # generate texts with input is psychological dimensions scores vector
    # for each dimenion, travel along the dimension range to generate text, while keep other dimensions as mean values (e.g. 0.0s)
    # examples of input:  [0,0,-2.std,0,0], [0,0,-1.std,0,0], [0,0,0,0,0], [0,0,+1.std,0,0], [0,0,+2.std,0,0]
    # the mean and std values for each dimenion will be retrieved from the training data files

    dimensions_name = data_df.columns[1:]
    if args.psych_variables == "big5":
        dimension_personality_dict = {"ope_z": "Openness", "con_z": "Conscientiousness", "ext_z": "Extroversion", "agr_z": "Agreeableness", "neu_z": "Neuroticism"}
    elif args.psych_variables == "dep":
        dimension_personality_dict = {"dep_score": "Depression"}
    elif args.psych_variables == "swl":
        dimension_personality_dict = {"swl": "Life-satisfaction"}
    else:
        assert args.psych_variables in ['big5', 'swl', 'dep'], "Currently, --psych_variables switch only supports: big5, dep, swl."
    dimensions_name = [dimension_personality_dict[item] for item in dimensions_name]
    means = np.mean(training_sentences_embeddings, axis = 0)
    stds = np.std(training_sentences_embeddings, axis = 0)


    # set parameters
    explore_std_range = [-args.std_range, args.std_range + 0.01]
    std_step_interval = args.generate_interval


    # generate text for each hidden dimension
    hidden_size = means.shape[0]
    for i in range(hidden_size):
    
        # print banner
        banner = "DIMENSION NUMBER {} ({}) : ".format(str(i), dimensions_name[i])
        print("="*len(banner))
        print(banner)   
        print("="*len(banner))

        # traveling std range and generating text
        for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):
        
            print("Sampling text at position Mean + ({})*Std:".format(round(std_position,2)))
            generated_samples = []   # avoid repeated generated_sample
            for _ in range(args.generate_num):     
                
                # sample embedding around embedding + std_position*stds[i]
                embedding_sample = np.copy(means)
                embedding_sample[i] = embedding_sample[i] + std_position*stds[i]

                # transform to tensor
                embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)

                # generate sentence
                # caveat: when generating, remove duplicated generation by checking newly generated text with already generated set
                generated_count = 0    
                while True:
                    generated_sample, decoder_attentions_sample = model.inference(prompting_text = args.prompting_text, sentence_embedding = embedding_sample, args = args, device = device)
                    generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
                    generated_count += 1
                    first_endoftext = generated_sample.find("<eos>") 
                    generated_sample_clean = generated_sample[:(first_endoftext + len("<eos>")) if first_endoftext>0 else len(generated_sample)]
                    generated_sample_clean = generated_sample_clean.replace("<bos>","<bos> ").replace("<eos>"," <eos>")
                    if (generated_sample_clean not in generated_samples) or generated_count >= 10:
                        generated_samples.append(generated_sample_clean)
                        break
                    
                # print generated sentence sample
                print("  ", generated_sample_clean)
            print("\n")

"""====================== MAIN FUNCTION ======================"""

# main function
def main():
    parser = argparse.ArgumentParser()

    # dataset/save path parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="Training data file, which also be used in inferencing for calculating mean and std values for each psychological variable.")                      
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be saved or loaded when inferencing.")
    parser.add_argument("--checkpoint_step", default=None, type=str, required=False,
                        help="Checkpoint step to load trained PEFT parameters from.")
    parser.add_argument("--psych_variables", default=None, type=str, required=True,
                        help="Psychological variables for generation. Currently supporting: big5, dep, swl.")
         
    # model parameters
    parser.add_argument("--model_name_or_path", default="google/gemma-2b", type=str,
                        help="The base model checkpoint for weights initialization.")
    parser.add_argument("--latent_size", default=-1, type=int, required=True,
                        help="Size of psychological variables vector. E.g., --latent_size=5 for big5; --latent_size=1 for dep/swl.")    
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.") 

    # generating parameters
    parser.add_argument("--generate_num", type=int, default=10,
                        help="Number of samples to generate for each variable value.")
    parser.add_argument("--generate_length", type=int, default=64,
                        help="Maximum tokens to generate.")
    parser.add_argument("--prompting_text", type=str, default=None,
                        help="Prompting for continuous generating.")
    parser.add_argument("--std_range", type=float, default=3.0,
                        help="Range of standard deviation to generate from. E.g.: from -3*std to +3*std.")
    parser.add_argument("--generate_interval", type=float, default=3.0,
                        help="Steps of standard deviation within the range of standard deviation. For example, setting --generate_interval=1.0 and --std_range to 3.0 will generate text at [-3*std, -2*std, -1*std, ..., 3*std].")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)

    # other parameters
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")

    # parsing parameters
    args = parser.parse_args()
    
    
    # =========== checking parameters and setting up  =========== #

    # setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")    # CHECK! make sure we use all 3 GPUs
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    # set seed
    set_seed(args)


    # =========== bulilding model and inferencing  =========== #
    # building model
    model = PsychAdapter(args.model_name_or_path, args.latent_size)
    
    # load from checkpoint model
    output_dir_basemodel = os.path.join(args.output_dir, 'base_model')
    output_dir_currentstep = os.path.join(args.output_dir, 'checkpoint-{}'.format(args.checkpoint_step))
    # load base model
    model.from_checkpoint(args, output_dir_basemodel)
    # load peft
    peft_model_id = output_dir_currentstep
    model = PeftModel.from_pretrained(model, peft_model_id)

    # send model to GPU
    model.to(args.device)    
    model.eval()

    # logging info
    print("Inference parameters %s", args)
    
    # check model size on memory
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print("Memory usage: {}MB".format(mem//(1024**2)))

    # testing inference
    args.model_config = model.model_config
    
    run_inference(model, args, device)    
    
if __name__ == "__main__":
    main()