# import libraries
import argparse
import logging
import os
import pickle
import random
import numpy as np
import torch
import pandas as pd


from torch.utils.data import DataLoader, Dataset, RandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import get_scheduler
from peft import get_peft_model, LoraConfig


from psychadapter import PsychAdapter

logger = logging.getLogger(__name__)

"""====================== METHODS DEFINITIONS ======================"""

def truncating_padding_sentence(tokens, block_size):
    if (len(tokens) > block_size):
        original_tokens_len = block_size
        tokens = tokens[:block_size]
    else:
        original_tokens_len = len(tokens)
        tokens = tokens + ["<pad>"]*(block_size - len(tokens))
    return tokens, original_tokens_len    

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, args):
        
        # reading data file
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_' + str(args.block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)       
                
            
            # reading file
            self.examples = []

            data_df = pd.read_csv(file_path, header = 0, index_col = False)
            for i, record in data_df.iterrows(): 

                # read data
                sentence_text = str(record["message"])
                sentence_embedding = np.array(record[2:].values).astype(float)

                # tokenize sentence
                sentence_tokenized = tokenizer.tokenize(sentence_text)

                # decoder_input
                decoder_input = [tokenizer.bos_token] + sentence_tokenized
                decoder_input, _ = truncating_padding_sentence(decoder_input, args.block_size)
                decoder_input = tokenizer.convert_tokens_to_ids(decoder_input)
                decoder_input = np.array(decoder_input)
                # decoder_output
                decoder_label = sentence_tokenized + [tokenizer.eos_token]
                decoder_label, _ = truncating_padding_sentence(decoder_label, args.block_size)
                decoder_label = tokenizer.convert_tokens_to_ids(decoder_label)
                decoder_label = np.array(decoder_label)

                # decoder_attention_mask
                decoder_attention_mask = 0

                # append to examples list
                training_sentence = dict({"sentence_embedding": sentence_embedding, "sentence_text": sentence_text, "decoder_input": decoder_input, "decoder_attention_mask": decoder_attention_mask, "decoder_label": decoder_label})  
                self.examples.append(training_sentence)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # print examples of training set
        logger.info("Examples of samples from training set:")
        for i in range(2):
            example = self.examples[i]
            logger.info("decoder_input: " + str(example["decoder_input"]))
            logger.info("decoder_label: " + str(example["decoder_label"]))           
            logger.info("decoder_input: " + str(tokenizer.decode(example["decoder_input"].tolist(), clean_up_tokenization_spaces=True))) 
            logger.info("decoder_label: " + str(tokenizer.decode(example["decoder_label"].tolist(), clean_up_tokenization_spaces=True)))
            logger.info("\n")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
    
def load_and_cache_examples(args, file_path, tokenizer):
    dataset = TextDataset(tokenizer, file_path=file_path, args=args)
    return dataset    

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def loss_fn(decoder_lm_logits, target, ignore_index):
    
    # negative Log Likelihood
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index) # this 'mean' is taking average across all predicted tokens: sum(crossentropyloss_each_position)/(batch_size * seq_length)
    # transform decoder_lm_logits from [batch_size, seq_length, vocab_size] => [batch_size * seq_length, vocab_size], target from [batch_size, sweq_length] => [batch_size * sweq_length]
    NLL_loss = loss_fct(decoder_lm_logits.view(-1, decoder_lm_logits.size(-1)), target.contiguous().view(-1))  

    return NLL_loss

def loss_perplexity_fn(decoder_lm_logits, target, ignore_index):
    
    # negative Log Likelihood
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index) # this 'mean' is taking average across all predicted tokens: sum(crossentropyloss_each_position)/(batch_size * seq_length)

    NLL_loss_batch = []
    perplexity_batch = []
    for i in range(decoder_lm_logits.shape[0]):
        NLL_loss_onesample = loss_fct(decoder_lm_logits[i], target[i])
        perplexity_onesample = torch.exp(NLL_loss_onesample)
        NLL_loss_batch.append(NLL_loss_onesample)
        perplexity_batch.append(perplexity_onesample)
    return NLL_loss_batch, perplexity_batch    

def save_checkpoint(model, args, loss_reports, global_step):
    # save peft checkpoints
    output_dir_currentstep = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir_currentstep):
        os.makedirs(output_dir_currentstep)
    model.module.save_pretrained(output_dir_currentstep) # save lora peft
    logger.info("Saving peft model checkpoint to %s", output_dir_currentstep)

    # save loggings
    output_dir_basemodel = os.path.join(args.output_dir, 'base_model')
    model.module.save_loggings(args, output_dir_basemodel, loss_reports)


"""====================== TRAIN/EVALUATE FUNCTION ======================"""

# train and evaluate function
def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """

    # summary writer
    tb_writer = SummaryWriter()
    
    # # DEBUGGING
    # logger.info("train_dataset: " + str(len(train_dataset)))
    # logger.info(train_dataset[0])
    # logger.info("train_batch_size: " + str(args.per_gpu_train_batch_size * max(1, args.n_gpu)))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},  
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # set optimizer and scheduler
    global_step = 0
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_scheduler(name="linear", optimizer = optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)    


    # run training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * 1)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  # added here for reproducibility (even between python 2 and 3)


    loss_report = []
    eval_loss_report = []
    eval_perplexity_loss_report = []
    eval_current_step = []
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
 
            sentence_embedding = batch["sentence_embedding"].float()
            decoder_input = batch["decoder_input"].long() 
            decoder_label = batch["decoder_label"].long()        

            # create decoder_attention_mask here instead of in the loop to save running time, the len is (decoder_input.shape[1] + 1), with 1 is for the past token 
            decoder_attention_mask = torch.tensor([[1]*(decoder_input.shape[1] + 1)]*decoder_input.shape[0], device = args.device)

            sentence_embedding = sentence_embedding.to(args.device)
            decoder_input = decoder_input.to(args.device)
            decoder_label = decoder_label.to(args.device)
            decoder_attention_mask = decoder_attention_mask.to(args.device)


            # forward pass
            decoder_lm_logits = model(sentence_embedding, decoder_input, decoder_attention_mask, args.device)


            # compute loss 
            NLL_loss = loss_fn(decoder_lm_logits, decoder_label, tokenizer.convert_tokens_to_ids(["<pad>"])[0]) 
            loss = NLL_loss


            # process loss across GPUs, batches then backwards
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps


            loss_report.append(loss.data.cpu().numpy())
            # run loss backward
            loss.backward()


            # accummulte enough step, step backward
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1


                # Logging 
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    averaged_loss = (tr_loss - logging_loss)/args.logging_steps
                    tb_writer.add_scalar('loss', averaged_loss, global_step)
                    logging_loss = tr_loss
                    logger.info("Current training step: " + str(global_step))
                    logger.info("Average current training loss of the latest {} steps: {}".format(str(args.logging_steps), str(averaged_loss)))  
                    

                # Save checkpoints
                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    # Evaluate checkpoint
                    if args.evaluate_during_training:
                        # set model to eval
                        model.eval()

                        # running train function
                        eval_loss, eval_perplexity = evaluate(args, eval_dataset, model, tokenizer)
                        eval_loss_report.append(eval_loss)
                        eval_perplexity_loss_report.append(eval_perplexity)
                        eval_current_step.append(global_step)

                        # set model to train
                        model.train()

                    # Save model (base model, peft checkpoints, loggings)
                    loss_reports = {"loss_report":loss_report, "eval_loss_report":eval_loss_report, "eval_perplexity_loss_report":eval_perplexity_loss_report, "eval_current_step":eval_current_step}
                    save_checkpoint(model, args, loss_reports, global_step)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # save final loss_reports
    loss_reports = {"loss_report":loss_report, "eval_loss_report":eval_loss_report, "eval_perplexity_loss_report":eval_perplexity_loss_report, "eval_current_step":eval_current_step}

    # close summary writer
    tb_writer.close()

    return global_step, tr_loss, loss_reports

def evaluate(args, eval_dataset, model, tokenizer):

    # set up
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # run evaluating
    logger.info("\n")
    logger.info("************************************")
    logger.info("***** Start running evaluating *****")
    logger.info("  Num examples = " + str(len(eval_dataset)))
    logger.info("  Instantaneous batch size per GPU = " + str(args.per_gpu_eval_batch_size))


    loss_report = []
    perplexity_report = []
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=False)):

        # extract input/output
        sentence_embedding = batch["sentence_embedding"].float()
        decoder_input = batch["decoder_input"].long() 
        decoder_label = batch["decoder_label"].long()        

        # create decoder_attention_mask here instead of in the loop to save running time, the len is (decoder_input.shape[1] + 1), with 1 is for the past token 
        decoder_attention_mask = torch.tensor([[1]*(decoder_input.shape[1] + 1)]*decoder_input.shape[0], device = args.device)

        # push to GPUs
        sentence_embedding = sentence_embedding.to(args.device)
        decoder_input = decoder_input.to(args.device)
        decoder_label = decoder_label.to(args.device)
        decoder_attention_mask = decoder_attention_mask.to(args.device)

        with torch.no_grad():
            model.eval()


            # forward pass (change and edit with VAE code)
            decoder_lm_logits = model(sentence_embedding, decoder_input, decoder_attention_mask, args.device)


            # compute loss  
            loss, perplexity = loss_perplexity_fn(decoder_lm_logits, decoder_label, tokenizer.convert_tokens_to_ids(["<pad>"])[0])    


            # process loss across GPUs, batches then backwards
            if args.n_gpu > 1:
                loss = torch.stack(loss)    # concatenate across all GPUs
                perplexity = torch.stack(perplexity)    # concatenate across all GPUs
            elif args.n_gpu == 1:
                loss = torch.tensor(loss)
                perplexity = torch.tensor(perplexity)


            loss_report.extend(loss.data.cpu().numpy())
            perplexity_report.extend(perplexity.data.cpu().numpy())

    # report results
    logger.info("=== Evaluating results ===")
    logger.info("Average evaluating loss: " + str(np.mean(loss_report)))
    logger.info("Average perplexity: " + str(round(np.mean(perplexity_report),3)))
    logger.info("***********************************")
    logger.info("***** End running evaluating *****")
    logger.info("\n")

    return np.mean(loss_report), np.mean(perplexity_report)


"""====================== MAIN FUNCTION ======================"""


# main function
def main():
    
    # =========== parameters parsing =========== #
    parser = argparse.ArgumentParser()

    # dataset and save/load paths arguments
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--start_step", type=int,
                        help="The checkpoint number.")    
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    
    # base model arguments
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--latent_size", default=-1, type=int, required=True,
                        help="Size of latent VAE layer.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.") 

    # training arguments
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")          
    parser.add_argument("--frozen_layers", type=str, default='None', 
                        help="Layers to be frozen while training.")
   
    # other arguments
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    
    # parsing parameters
    args = parser.parse_args()
    
    
    # =========== checking parameters and setting up  =========== #
    # checking parameters validity
    if args.do_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
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


    # =========== bulilding model and training/evaluating  =========== #

    # building model
    model = PsychAdapter(args.model_name_or_path, args.latent_size)

    # initialize / load from checkpoint model
    if args.do_train:
        # initialize model with pretrained decoder model and randomly initialized transformation_matrix
        model.initialize_model(args)    

        # save base model
        output_dir_basemodel = os.path.join(args.output_dir, 'base_model')
        if not os.path.exists(output_dir_basemodel):
            os.makedirs(output_dir_basemodel)
        model.save_basemodel(args, output_dir_basemodel) # save decoder and transformation_matrix

    if args.block_size <= 0:  # modify args.block_size variable
        args.block_size = model.tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, model.tokenizer.max_len_single_sentence)

    # report model's architecture and size
    pytorch_total_params = sum(p.numel() for p in model.parameters()) 
    logger.info("Model architecture: ")
    logger.info(model)
    logger.info("Number of parameters of transform_matrix: " + str(sum(p.numel() for p in model.transform_matrix.parameters()) ))
    logger.info("Number of parameters of base_decoder: " + str(sum(p.numel() for p in model.decoder.parameters()) ))
    logger.info("Number of parameters total: " + str(pytorch_total_params))

    # setup PEFT
    if "gpt2" in args.model_name_or_path:
        peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["c_proj"])
    else:
        peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_config)
    logger.info("PEFT trainable parameters: ")
    model.print_trainable_parameters()
    
    # send model to GPU
    model.to(args.device)

    # set up data parallel
    model = torch.nn.DataParallel(model) 

    # training
    if args.do_train:
            
        # freeze layers if appllicable
        if args.frozen_layers is not None:
            frozen_layers = args.frozen_layers.split(" ")
            for name, param in model.named_parameters():
                if any(".{}.".format(str(frozen_layer)) in name for frozen_layer in frozen_layers):
                    logger.info("frozen params: " + name)
                    param.requires_grad = False
            
            
        # load training dataset
        args.model_config = model.module.model_config
        train_dataset = load_and_cache_examples(args, args.train_data_file, model.module.tokenizer)
        if args.evaluate_during_training:
            eval_dataset = load_and_cache_examples(args, args.eval_data_file, model.module.tokenizer)
        else:
            eval_dataset = None

        # set model to train
        model.train()

        # running train function
        train(args, train_dataset, eval_dataset, model, model.module.tokenizer)

        # good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))        

if __name__ == "__main__":
    main()        



