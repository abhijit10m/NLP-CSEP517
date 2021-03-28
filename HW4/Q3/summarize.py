from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from sacremoses import MosesTokenizer, MosesDetokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import random
import numpy as np

md = MosesDetokenizer(lang='en')





############################################################
############## START OF SETUP ##############################
############################################################




model = GPT2LMHeadModel.from_pretrained("gpt2")

encoder = GPT2Tokenizer.from_pretrained('gpt2')


from transformers import BartForConditionalGeneration, BartTokenizer, top_k_top_p_filtering
import torch.nn.functional as F

end_tok = encoder.encode('<|endoftext|>')[0]


def generate(model, input_ids, prefix_past, min_gen_len = 30, gen_len = 30):
    generation = []

    past_key_values = prefix_past

    for j in range(gen_len):

        outputs = model(input_ids = input_ids, past_key_values=past_key_values )

        logits = outputs.logits.cpu()[0,-1]

        logits[198] = -float("Inf")
        logits[628] = -float("Inf")

        if j < min_gen_len:
            logits[end_tok] = -float("Inf")

        logits = top_k_top_p_filtering(logits.view(1,-1), top_p=0.5)

        
        next_tok = torch.multinomial(F.softmax(logits, dim=1).cpu(), num_samples=1)[0,0]

        if next_tok == end_tok:
            print('break')
            break

        input_ids = torch.tensor([next_tok]).view(1,-1).to(device)
        generation += [next_tok.tolist()]
        past_key_values = outputs.past_key_values


    return generation




def get_inputs_embeds(model, input_ids):
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    batch_size = input_ids.shape[0]

    inputs_embeds = model.transformer.wte(input_ids)
    return inputs_embeds



from tqdm import tqdm


gen_delim = encoder.encode('\nSummary:\n')
gen_delim
gen_end = encoder.encode('<|endoftext|>')



data_train = list(np.load('train.npy', allow_pickle=True)[0])
data_test = list(np.load('test.npy', allow_pickle=True)[0])



def get_index(l, sl):
    for i in range(len(l)):
        if l[i:i + len(sl)] == sl:
            return i
    return -1










flatten = lambda t: [item for sublist in t for item in sublist]

def collate_data(data):
    max_len = max([len(d) for d in data])
    
    X = torch.zeros(len(data), max_len).long()
    Y = torch.zeros(len(data), max_len).long()
    
    Y[:,:] = -100
    

    for i, d in enumerate(data):
        #assert(gen_delim in d)

        
        
        X[i,:len(d)] = torch.tensor(d, requires_grad=False).long()
        Y[i,:len(d)] = torch.tensor(d, requires_grad=False).long()
        
        ## only predict after generation start
        Y[i,:get_index(d, gen_delim)  +len(gen_delim)] = -100
        
        
    return X, Y


############################################################
############## END OF SETUP ################################
############################################################


############################################################
############## START OF TRAINING SECTION####################
############################################################

data = data_train[:500]
num_epochs = 3
effective_batch_size = 4
batch_size = 4
num_warmup_steps=2
prefix_len = 100
lr = 1e-3

assert((effective_batch_size % batch_size) == 0)

grad_acc_steps = effective_batch_size / batch_size

batches = [data[i*batch_size:(i+1)*batch_size] for i in range(int(len(data)/batch_size))]

random.shuffle(data)

device = model.transformer.wte.weight.device

output = model(input_ids = torch.tensor([49736]*prefix_len).view(1,-1).to(device))
prefix_past = tuple([tuple([t.clone().detach().requires_grad_(True) for t in v]) for v in output.past_key_values])



optimizer = AdamW(flatten(prefix_past), lr=lr)



num_train_steps = len(batches)*num_epochs/grad_acc_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

model.eval()
device = model.transformer.wte.weight.device

losses = []
log_steps = 10
print('{} batches\n====='.format(len(batches)))



############################################################
############## START OF Generation SECTION##################
############################################################
    
def generate_file(epoch):
    generations = []
    print('|| Generating summaries ||')
    for d in tqdm(data_test):
        end_tok = encoder.encode('<|endoftext|>')[0]
        input_ids = torch.tensor(d[:get_index(d, gen_delim)  +len(gen_delim)]).view(1,-1).to(device)
        generations += [encoder.decode(generate(model, input_ids, prefix_past))]
        
        
    import os

    files = os.listdir('../generations/')

    for i in range(20):
        fname = 'gpt2_{}_epoch_{}.txt'.format(i, epoch) 
        if fname not in files:
            break


    with open('../generations/' + fname,'w') as f:
        for g in generations:
            f.write(g + '\n')
############################################################
############## END OF GENERATION SECTION####################
############################################################

###############################################
################## Q3.1 #######################
##
## Your goal in this question is to adapt the code
## below to iterate over the whole training set (not just)
## one example (right now, it's just doing the first
## example in the dataset)
##
## And to do this num_epochs times 
##
## !!! CODE GOES HERE

for epoch in range(num_epochs):
    # for batch in batches:
    for i, batch in tqdm(enumerate(batches)):
        
        ### BELOW IS THE TRAINING BLOCK. DO NOT CHANGE
        X, Y = collate_data(batch)
        X = X.to(device)
        Y = Y.to(device)
        output = model( input_ids = X, labels=Y,past_key_values=tuple([tuple([t.expand(X.shape[0],-1,-1,-1) for t in v]) for v in prefix_past]))
        loss = output.loss
        losses += [loss.tolist()]
        if i % log_steps == 0:
            print('iter {}: avg loss {}'.format(i, sum(losses[-20:])/ len(losses[-20:]) ))
        loss = loss/grad_acc_steps
        loss.backward()
        if i % grad_acc_steps == 0:
            optimizer.step()
            scheduler.step()
        torch.cuda.empty_cache()
    print('epoch: {}'.format(epoch))
    generate_file(epoch)
### END OF TRAINING BLOCK

## END OF Q3.1 
#############################
    

    
############################################################
############## END OF TRAINING SECTION######################
############################################################
    
    
