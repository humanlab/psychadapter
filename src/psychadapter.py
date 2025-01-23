# import libraries
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import logging
import torch.nn.functional as F
import os
import pickle

logger = logging.getLogger(__name__)

# ===================== classes of model ===================== #

class PsychAdapter(nn.Module):

    def __init__(self, model_name_or_path, latent_size):
        super(PsychAdapter, self).__init__()

        # set up transformation matrix and decoder
        self.model_config = AutoConfig.from_pretrained(model_name_or_path, cache_dir = None)
        self.transform_matrix = nn.Linear(latent_size, self.model_config.num_hidden_layers * 2 * self.model_config.num_key_value_heads * self.model_config.head_dim)
        self.tokenizer = None
        self.decoder = None

        # set up model_config
        self.model_config.output_hidden_states = True
        self.model_config.use_cache = True
        self.model_config.output_attentions = True


    def initialize_model(self, args):

        # load pretrained model and tokenizer for the encoder and decoder
        decoder_path = args.model_name_or_path   
        tokenizer_path = args.model_name_or_path
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_path, from_tf=bool('.ckpt' in decoder_path), config=self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case, add_prefix_space=True)

        # special tokens from tokenizer
        # <bos><eos><pad>
        bos_token_id = self.model_config.bos_token_id
        eos_token_id = self.model_config.eos_token_id
        pad_token_id = self.model_config.pad_token_id


    def forward(self, embeddings, decoder_input_ids, decoder_attention_mask, device):

        # batch_size
        batch_size = embeddings.shape[0]

        # transform input vector to transformers model hidden's size
        # past_key_values: Tuple of tuple(torch.FloatTensor) of length config.n_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head))
        transformed_embeddings = self.transform_matrix(embeddings) # [batch_size, (self.model_config.num_hidden_layers * 2 * self.model_config.num_key_value_heads * self.model_config.head_dim)]
        transformed_embeddings = transformed_embeddings.reshape([batch_size, self.model_config.num_hidden_layers, 2, self.model_config.num_key_value_heads, 1, self.model_config.head_dim])
        transformed_embeddings = torch.transpose(transformed_embeddings,0,1).contiguous()
        transformed_embeddings = torch.transpose(transformed_embeddings,1,2).contiguous()

        # decoder
        past = transformed_embeddings

        # decoder forward pass
        decoder_lm_logits, decoder_presents, decoder_hidden_states, decoder_attentions = self.decoder(input_ids = decoder_input_ids, past_key_values = past, attention_mask = decoder_attention_mask, return_dict=False)

        return decoder_lm_logits


    def inference(self, prompting_text = None, sentence_embedding = None, args = None, device = None):

        # make sure batch_size = 1
        batch_size = sentence_embedding.shape[0]
        assert batch_size == 1

        # transform input vector to transformers model hidden's size
        # past_key_values: Tuple of tuple(torch.FloatTensor) of length config.n_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head))
        transformed_embeddings = self.transform_matrix(sentence_embedding) # [batch_size, (self.model_config.num_hidden_layers * 2 * self.model_config.num_key_value_heads * self.model_config.head_dim)]
        transformed_embeddings = transformed_embeddings.reshape([batch_size, self.model_config.num_hidden_layers, 2, self.model_config.num_key_value_heads, 1, self.model_config.head_dim])
        transformed_embeddings = torch.transpose(transformed_embeddings,0,1).contiguous()
        transformed_embeddings = torch.transpose(transformed_embeddings,1,2).contiguous()

        # decoder
        if prompting_text is None:
            decoder_input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids("<bos>")]*batch_size, device = device).long().reshape(batch_size,1)
        else:
            prompting_text_tokens = "<bos>" + prompting_text.strip()
            prompting_text_encoded = self.tokenizer.encode(prompting_text_tokens, add_special_tokens = False)
            decoder_input_ids = torch.tensor(prompting_text_encoded*batch_size, device = device).long().reshape(batch_size,len(prompting_text_encoded))
        past = transformed_embeddings

        # generate tokens
        generated = decoder_input_ids
        for _ in range(args.generate_length):

            # compute attention mask
            decoder_attention_mask = torch.tensor([[1]*(generated.shape[1] + 1)]*generated.shape[0], device = device)

            # decoder forward pass
            decoder_lm_logits, decoder_presents, decoder_hidden_states, decoder_attentions = self.decoder(input_ids = generated, past_key_values = past, attention_mask = decoder_attention_mask, return_dict=False)

            # sample from vocabulary
            decoder_lm_logits = decoder_lm_logits[:,-1,:]

            filtered_decoder_lm_logits = top_k_top_p_filtering(decoder_lm_logits, top_k=args.top_k, top_p=args.top_p)
            if args.temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_decoder_lm_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_decoder_lm_logits/args.temperature, dim=-1), num_samples=1)                
            generated = torch.cat((generated, next_token), dim=1)
    
        return generated, decoder_attentions


    def save_basemodel(self, args, output_dir):

        # set up output_dir to save sub-models
        output_dir_decoder = output_dir + "/decoder/"
        output_dir_tokenizer = output_dir + "/tokenizer/"
        output_dir_transform_matrix = output_dir + "/transform_matrix/"
        if not os.path.exists(output_dir_decoder):
            os.makedirs(output_dir_decoder)            
        if not os.path.exists(output_dir_tokenizer):
            os.makedirs(output_dir_tokenizer)
        if not os.path.exists(output_dir_transform_matrix):
            os.makedirs(output_dir_transform_matrix)
        output_dir_transform_matrix = output_dir_transform_matrix + "/transform_matrix.weights"    

        # save model
        self.decoder.save_pretrained(output_dir_decoder)
        self.tokenizer.save_pretrained(output_dir_tokenizer)
        torch.save(self.transform_matrix.state_dict(),output_dir_transform_matrix)    

        return


    def save_loggings(self, args, output_dir, loss_reports):

        # save training args and loss record
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)
        loss_reports_file = open(output_dir + "/loss_reports.pkl", "wb")
        pickle.dump(loss_reports, loss_reports_file)
        
        return


    def from_checkpoint(self, args, output_dir):
        
        model_dir = output_dir

        # loading from pre-trained
        decoder_path = model_dir + "/decoder/"
        tokenizer_path = model_dir + "/tokenizer/"
        transform_matrix_path = model_dir + "/transform_matrix/transform_matrix.weights"
        logger.info("model_config: " + str(self.model_config))
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_path, from_tf=bool('.ckpt' in decoder_path), config=self.model_config, ignore_mismatched_sizes = False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)
        if torch.cuda.is_available():
            self.transform_matrix.load_state_dict(torch.load(transform_matrix_path))
        else:
            self.transform_matrix.load_state_dict(torch.load(transform_matrix_path, map_location=torch.device('cpu')))

        # set up for evaluating
        self.decoder.eval()
        self.transform_matrix.eval()

        # load training args
        training_args = torch.load(os.path.join(model_dir, 'training_args.bin'))

        return 

# ===================== other methods ===================== #
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits



