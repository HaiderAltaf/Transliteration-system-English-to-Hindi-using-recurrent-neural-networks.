#!/usr/bin/env python
# coding: utf-8

# ## Requirements

# In[1]:


import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
from prettytable import PrettyTable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[45]:


#pip install prettytable


# ## Argparse

# In[19]:


# Using argparse, I have define the arguments and options that my program accepts,
# and argparse will run the code, pass arguments from command line and 
# automatically generate help messages. I have given the defaults values for 
# all the arguments, so code can be run without passing any arguments.
# lastly returning the arguments to be used in the running of the code.

import argparse

parser = argparse.ArgumentParser(description="Stores all the hyperpamaters for the model.")
parser.add_argument("-wp", "--wandb_project",default="cs6910_assignment 3 new" ,type=str,
                    help="Enter the Name of your Wandb Project")
parser.add_argument("-we", "--wandb_entity", default="am22s020",type=str,
                    help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument("-ws", "--wandb_sweep", default="False", type=bool,
                    help="If you want to run wandb sweep then give True")
parser.add_argument("-e", "--epochs",default="1", type=int, choices=[1, 5, 10],
                    help="Number of epochs to train neural network.")
parser.add_argument("-hs", "--hidden_size",default="256", type=int, help="no. of neurons in the hidden layer of the N/W")
parser.add_argument("-c", "--cell_type",default="lstm", type=str, choices=["lstm", "gru", "rnn"])
parser.add_argument("-nl", "--num_layers",default="2", type=int, 
                    choices=[2, 3, 4], help="number of recurrent layers")
parser.add_argument("-ems", "--embedding_size", default="256", type=int, choices=[64, 128, 256])
parser.add_argument("-bd", "--bi_directional", default="True", type=bool)

args = parser.parse_args()

wandb_project = args.wandb_project
wandb_entity = args.wandb_entity
wandb_sweep = args.wandb_sweep
num_epochs = args.epochs
hidden_size = args.hidden_size
cell_type = args.cell_type
num_layers = args.num_layers
embedding_size = args.embedding_size
bi_directional = args.bi_directional

print("wandb_project :", wandb_project , "wandb_entity: ", wandb_entity,"wandb_sweep: ",wandb_sweep,
      "epochs: ",num_epochs,"hidden_size: ",hidden_size, "cell_type: ", cell_type,
      "num_layers: ",num_layers,"embedding_size: ",embedding_size, 
      "bi_directional: ", bi_directional)


# ## Preparing the datasets

# In[2]:


class Vocabulary():
    """
    This class(Vocabulary), builds a character-level vocabulary for a given list of words.
    It initializes the vocabulary with four special tokens (PAD, SOW, EOW, and UNK) and creates
    two dictionaries (stoi and itos) to map characters to integers and vice versa.
    Tokenizer: Tokenizes a given text into individual characters.
    build_vocabulary(): Takes a list of words and adds each unique character 
    to the vocabulary, along with a unique integer ID.
    numericalize(): Converts a given text into a list of integers, where each 
    integer corresponds to the ID of a character in the vocabulary.
    
    """
    def __init__(self):
        self.itos = {0:"<PAD>",1:"<SOW>",2:"<EOW>",3:"<UNK>"}
        self.stoi = {"<PAD>":0,"<SOW>":1,"<EOW>":2,"<UNK>":3}
        #self.freq_threshold = freq_threshold
    
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer(text):
        return [*text]
    
    def build_vocabulary(self, word_list):
        char_list = []
        idx = 4
        
        for word in word_list:
            for char in self.tokenizer(word):
                if char not in char_list:
                    char_list.append(char)
                    self.stoi[char] = idx
                    self.itos[idx] = char
                    idx+=1
                    
                    
    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]
                 


# In[3]:


class aksharantar(Dataset):
    """
    This class used to process text data for a machine translation task.
    root_dir: the root directory where the data is stored
    out_lang: the target language for translation 
    dataset_type: either "train", "test", or "val" indicating which dataset is being used.
    After loadinf data __init__() builds the vocabulary for each language by adding all unique characters in 
    the language's text data to the corresponding Vocabulary object.
    The __getitem__() method takes an index and returns the numericalized form of the corresponding input 
    and output sentences.
    It tokenizes each sentence into characters and adds special start-of-word (<SOW>) and end-of-word (<EOW>) 
    tokens to the beginning and end of the numericalized output sentence.
    Finally, it returns PyTorch tensors of the numericalized input and output sentences.
    
    """
        
    def __init__(self, root_dir, out_lang, dataset_type): 
        
        # Read file
        self.file_name = out_lang + "_" + dataset_type + ".csv"
        self.file_dir = os.path.join(root_dir, out_lang, self.file_name)
        self.df = pd.read_csv(self.file_dir, names = ["latin", "hindi"])
        
        # Get columns of input and output language
        self.latin = self.df["latin"]
        self.hindi = self.df["hindi"]
        
        # Initialize vocabulary and build vocab
        self.vocab_eng = Vocabulary()
        self.vocab_eng.build_vocabulary(self.latin.tolist())
        
        # Initialize vocabulary and build vocab
        self.vocab_hin = Vocabulary()
        self.vocab_hin.build_vocabulary(self.hindi.tolist())
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        latin = self.latin[index]
        hindi = self.hindi[index]
        
        numericalized_hindi = [self.vocab_hin.stoi["<SOW>"]]
        numericalized_hindi += self.vocab_hin.numericalize(hindi)
        numericalized_hindi.append(self.vocab_hin.stoi["<EOW>"])
        
        numericalized_latin = [self.vocab_eng.stoi["<SOW>"]]
        numericalized_latin += self.vocab_eng.numericalize(latin)
        numericalized_latin.append(self.vocab_eng.stoi["<EOW>"])
        
        return torch.tensor(numericalized_latin), torch.tensor(numericalized_hindi) 
               
        


# In[4]:


class MyCollate:
    """
    This class is used to collate the data items into batches for DataLoader. 
    It takes two arguments, pad_idx_eng and pad_idx_hin, which are the index of the <PAD> token
    in the English and Hindi vocabularies respectively.
    
    
    """
    def __init__(self, pad_idx_eng, pad_idx_hin):
        self.pad_idx_eng = pad_idx_eng
        self.pad_idx_hin = pad_idx_hin
        
    def __call__(self, batch):
        inputs = [item[0] for item in batch]
        inputs = pad_sequence(inputs, batch_first=False, padding_value=self.pad_idx_eng)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx_hin)
        
        return inputs, targets
        
        


# In[5]:


def get_loader(root_dir, out_lang, dataset_type, batch_size, pin_memory=True ):
    """
    This class returns a PyTorch DataLoader object and a custom dataset object. 
    The DataLoader object loads the data in batches.
    
    """
    
    dataset = aksharantar(root_dir, out_lang, dataset_type)
    
    pad_idx_eng = dataset.vocab_eng.stoi["<PAD>"]
    pad_idx_hin = dataset.vocab_hin.stoi["<PAD>"]
    
    loader = DataLoader(dataset=dataset,batch_size=batch_size,
                       pin_memory=pin_memory,
                       collate_fn=MyCollate(pad_idx_eng=pad_idx_eng, pad_idx_hin=pad_idx_hin),
                       shuffle=True)
    return loader, dataset


# ## Getting the model Ready

# In[6]:


class Encoder(nn.Module):
    """
    This code defines an Encoder class for a sequence-to-sequence model.
    
    embedding_size:The embedding size is the dimensionality of the dense vector representation
    hidden_size: the number of units or neurons in the hidden layer of the network.
    num_layers: The number of layers in an RNN refers to the number of recurrent layers that are
    stacked on top of each other to process sequential input.
    bi_directional : input sequence to be processed in both forward and backward directions.
    
    The class takes in an input size, embedding size, hidden size, 
    number of layers, dropout rate, cell type (GRU, LSTM, or RNN), 
    and whether the network is bidirectional. The forward method takes in 
    an input tensor x, applies dropout to its embedded representation, and 
    passes it through a GRU, LSTM, or RNN layer depending on the cell type specified. 
    The final hidden states of the layer(s) are returned.
    
    """
    #input_size represents the dimensionality of the 
    #encoder's input space, indicating the number of possible input tokens or
    #categories that the coder can generate.
    
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, cell_type, bi_directional):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=bi_directional)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p,bidirectional=bi_directional)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=p,bidirectional=bi_directional)

    def forward(self, x):
        # x, shape=(seq_length, N)
        embedding = self.dropout(self.embedding(x))
        # embedding shape = (seq_length, N,embedding_size )
        
        if self.cell_type == 'gru':
            _, hidden = self.gru(embedding)
            return hidden
        
        if self.cell_type == 'lstm':
            _, (hidden, cell) = self.lstm(embedding)
            return hidden, cell
        
        if self.cell_type == 'rnn':
            _, hidden = self.rnn(embedding)
            return hidden
          
    # This method is called at the beginning of each new input sequence
    # to reset the hidden state.
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)


# In[7]:


class Decoder(nn.Module):
    """
    This code defines the Decoder class, which is responsible for decoding the encoded input sequence
    and generating the target sequence. 
    The method first unsqueezes x to add a batch dimension and then applies dropout to the embedding layer. 
    It then passes the embedded input sequence through the decoder's RNN layer, 
    which can be either GRU, LSTM, or RNN.
    Then passes the output through a linear layer to get the predictions, which are returned 
    along with the hidden and cell states.
    Finally, the method squeezes the predictions tensor to remove the batch dimension before returning 
    the predictions and hidden/cell states.
    
    """
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers,
                 p, cell_type, bi_directional ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p,bidirectional=bi_directional )
        self.lstm = nn.LSTM(embedding_size, hidden_size,num_layers, dropout=p, bidirectional=bi_directional)
        self.rnn = nn.RNN(embedding_size, hidden_size,num_layers, dropout=p, bidirectional=bi_directional)
        if bi_directional:
            self.fc = nn.Linear(2*hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)        
        
    def forward(self, x, hidden, cell):
        # x, shape=(N) but we want (1, N)
        x = x.unsqueeze(0)
        
        embedding = self.dropout(self.embedding(x))
        # embedding shape = (1, N,embedding_size )
        
        if self.cell_type == 'gru':
            outputs, hidden = self.gru(embedding, hidden)
            #shape of output (1,N,hidden_size)
            
        if self.cell_type == 'lstm':
            outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
            
        if self.cell_type == 'rnn':
            outputs, hidden = self.rnn(embedding, hidden)
            
        predictions = self.fc(outputs)
        # shape of predictions = (1, N, length_of_vocabs)
        
        predictions = predictions.squeeze(0)
        
        if self.cell_type == 'lstm':
            return predictions, hidden, cell
        else:
            return predictions, hidden
    


# In[8]:


class Seq2Seq(nn.Module):
    """
    This class have functions which takes words as input and target words to find the 
    predictions using the model build in the forward function.
    This function uses the encoder and decoder formed earlier.
    
    """
    def __init__(self, encoder, decoder, cell_type):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type
        
    def forward(self, word_input, word_target, teacher_force_ratio=0.5):
        
        batch_size = word_input.shape[1]
        target_length = word_target.shape[0]
        
        outputs = torch.zeros(target_length, batch_size, len(train_data.vocab_hin)).to(device)
        
        if self.cell_type == 'lstm':
             hidden, cell = self.encoder(word_input)
        else:
            hidden = self.encoder(word_input)
            
        # grab start token
        x= word_target[0]
        
        for t in range(1, target_length):
            if self.cell_type == "lstm":
                output, hidden, cell = self.decoder(x, hidden, cell)
            else:
                output, hidden = self.decoder(x, hidden, 0)
                
            outputs[t] = output
            
            best_pred = output.argmax(1)
            
            x = word_target[t] if random.random() < teacher_force_ratio else best_pred
            
        return outputs
    


# ## Functions to find accuracy and print and save outputs

# In[9]:


def predict(model, input_list, cell_type, max_length=30):
    
    '''
    max_length: The maximum length of latin input.
    The purpose of this function is to accept a list of characters in the input 
    language and then provide a list of characters in the output language.
    cell_type: to use which among lstm, rnn or gru cell
    
    
    '''
    
    # Making the indexes of the input according to the training data vocabulary
    # Because the index2str dicts of train data and val/test datasets are diffent
    
    input_word = [train_data.vocab_eng.stoi[char] for char in input_list]
    input_word = torch.LongTensor(input_word)

    # Input word is of shape (seq_length) but we want it to be (seq_length, 1) where 1 represents batch size
    input_word = input_word.view(input_word.shape[0],1).to(device)
    
    start_token_index = 1
    end_token_index = 2
   
    # Encoder
    with torch.no_grad():
        if model.cell_type == "lstm":
            hidden, cell = model.encoder(input_word)
        else:
            hidden = model.encoder(input_word)
    
    # Add start token to outputs
    outputs = [start_token_index]

    for _ in range(max_length):
        prev_char = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            if model.cell_type == "lstm":
                output, hidden, cell = model.decoder(prev_char, hidden, cell)
            else:
                output, hidden = model.decoder(prev_char, hidden, 0)
            
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == end_token_index:
            break
    
    # Convert outputs to character list
    prediction = [train_data.vocab_hin.itos[index] for index in outputs]
    
    return prediction


# In[10]:


def accuracy(dataset, model, cell_type):
    
    """
    This function will comapre the prediction given by the predict function and the target output.
    I will do word by word, so may take little more time.
    
    """
    
    # Initializing the count
    correct_count = 0
    # Number of data in our dataset
    words_count = len(dataset)
    
    for i in range(words_count):
           
        char_input = [dataset.vocab_eng.itos[index] for index in dataset[i][0].tolist()]
        
        prediction = predict(model, char_input, cell_type)
        
        actual_word = [dataset.vocab_hin.itos[index] for index in dataset[i][1].tolist()]
        
        if prediction == actual_word:
            correct_count+=1
            
            
    return 100*(correct_count/words_count)


# In[11]:


def prediction_csv(dataset, model, cell_type):
    
    """
    This function will generate the csv file containig the inputs their 
    predictions and corresponding target word.
    model: Trained model
    
    """
    
    correct_count = 0
    words_count = len(dataset)
    
    list_of_words = []
    
    for i in range(words_count):
        list1 = []
        
        char_input = [dataset.vocab_eng.itos[index] for index in dataset[i][0].tolist()]
        
        input_string = ''.join(char_input[1:len(char_input)-1])
        list1.append(input_string)
        
        prediction = predict(model, char_input, cell_type)
        
        pred_string = ''.join(prediction[1:len(prediction)-1])
        list1.append(pred_string)
        
        actual_word = [dataset.vocab_hin.itos[index] for index in dataset[i][1].tolist()]
        
        target_string = ''.join(actual_word[1:len(actual_word)-1])
        list1.append(target_string)
        list_of_words.append(list1)
        
        if prediction == actual_word:
            correct_count+=1
            
    with open('predictions_vanilla.csv', mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
    
        header = ["Input", "Prediction","Target"]
    
        # Write header row
        writer.writerow(header)
        
        for i in range(words_count):
            writer.writerow(list_of_words[i])
            


# In[12]:


def print_prediction(dataset, model, cell_type):
    
    """
    This function will generate the print the inputs, their 
    predictions and corresponding target word.
    model: Trained model
    
    """
    
    words_count = len(dataset)
    
    # For better looks of printed elements
    table = PrettyTable()
    table.field_names = ["Input", "Prediction", "Target"]
    
    #Creating the dataframe to store the 3 values
    df = pd.DataFrame(columns=['Input', 'prediction', 'Target'])
    
    for i in range(16):
        list1 = []
        
        # changing the word into list of characters.
        char_input = [dataset.vocab_eng.itos[index] for index in dataset[i][0].tolist()]
        
        input_string = ''.join(char_input[1:len(char_input)-1])
        list1.append(input_string)
        
        # getting the prediction using the predict function
        prediction = predict(model, char_input, cell_type)
        
        pred_string = ''.join(prediction[1:len(prediction)-1])
        list1.append(pred_string)
        
        # Preparing target word
        actual_word = [dataset.vocab_hin.itos[index] for index in dataset[i][1].tolist()]
        
        target_string = ''.join(actual_word[1:len(actual_word)-1])
        list1.append(target_string)

        # appending all the three vaues in their respective columns
        df = df.append({'Input': list1[0], 'prediction': list1[1], 'Target': list1[2]}, ignore_index=True)
    
    # Adding the content of dataframe in the the table
    for row in df.itertuples(index=False):
        table.add_row(row)
    
    print(table)
        


# In[15]:


def train(num_epochs, learning_rate, batch_size, load_model, 
         input_size_encoder, input_size_decoder, output_size,
         encoder_embedding_size, decoder_embedding_size,
         hidden_size, num_layers, enc_dropout, de_dropout):
    
    """
    This function is created to train the Seq2Seq model manually(without wandb).
    It takes the all the arguments needed for the encoder, decoder and Seq2seq model.
    Using this function we can test our model on test dataset, just uncomment the relevant line 
    commented in the lower part of the code.
    We can also generate prediction_vanilla csv file just by uncomment the 
    second last commented part of this code.
    We can also print the prediction by uncommenting the last part
    
    """
   
    
    # Importing the Encoder class
    encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                         hidden_size, num_layers, enc_dropout, cell_type,
                          bi_directional).to(device)
    
    # Importing the Decoder class
    decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                         hidden_size, output_size, num_layers, dec_dropout, 
                          cell_type ,bi_directional).to(device)

    # Preparing the model
    model = Seq2Seq(encoder_net, decoder_net, cell_type).to(device)
    
    # Importing the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    pad_index = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

    if load_model:
        load_checkpoint(torch.load('my_checkpoint.pth.ptar'),model, optimizer)
    
    print("Training the model...")
    for epoch in range(num_epochs):
        print('Epoch: ', epoch+1, '/', num_epochs)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            
            input_word = inputs.to(device)
            target_word = targets.to(device)

            output = model(input_word, target_word)
            # output shape: (target_len, batch_size, output_vocab_size)
            
            output = output[1:].reshape(-1, output.shape[2])
            target_word = target_word[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target_word)

            loss.backward()

            # To handle large gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
        print("Training Loss: ", loss.item())
        model.eval()
        print("Finding accuracy of the model...")
        train_accu =  accuracy(train_data, model, cell_type)
        valid_accu =  accuracy(valid_data, model, cell_type)
        model.train()

        print("valid accuracy: ", valid_accu)
        print("train accuracy: ", train_accu)
                        
        
    ##=> For testing the model uncomment the code
    #model.eval()        
    #test_accur =  accuracy(test_data, model, cell_type) 

    ##=> Uncomment  prediction_csv only if you want to generate the csv file of predictions
    #prediction_csv(test_data, model, cell_type)  
    #model.train()  
    
#     print("Printing the predictions of Input words:")
#     print()
#     model.eval()
#     print_prediction(test_data, model, cell_type)
#     model.train()    
    
    


# In[16]:


# # Data Uploading
# You can change the directory according to your data location
# out_lang: Choose which output language you want transliteration.
# 'hin':Hindi, 'urd':Urdu, 'tel':Telgu etc
root_dir = r'C:\Users\HICLIPS-ASK\aksharantar_sampled'
out_lang = 'hin'
batch_size = 64
train_loader, train_data = get_loader(root_dir, out_lang, 'train', batch_size=batch_size, pin_memory=True )
valid_loader, valid_data = get_loader(root_dir, out_lang, 'valid', batch_size=batch_size, pin_memory=True)
test_loader, test_data = get_loader(root_dir, out_lang, 'test', batch_size=batch_size, pin_memory=True)


# In[17]:


#wandb_sweep = False


# In[47]:


# To run manually Uncomment the 'wandb_sweep = False' in above line
if wandb_sweep == False:
    ## Giving the argument values for manual training
    num_epochs =1
    learning_rate = 0.001
    load_model = False
    input_size_encoder = len(train_data.vocab_eng)
    input_size_decoder = len(train_data.vocab_hin)
    output_size = len(train_data.vocab_hin)
    encoder_embedding_size = 256
    decoder_embedding_size = 256
    hidden_size = 256
    num_layers = 2
    enc_dropout = 0.2
    dec_dropout = 0.2
    load_model = False
    cell_type = 'lstm'
    bi_directional = True

    ## Training the model
    train(num_epochs, learning_rate, batch_size, load_model, 
             input_size_encoder, input_size_decoder, output_size,
             encoder_embedding_size, decoder_embedding_size,
             hidden_size, num_layers, enc_dropout, dec_dropout)


# ## Train with Wandb

# In[20]:


project_name = "Assignment 3"
entity_name = "am22s020"
import wandb


# In[21]:


def train_with_wandb():


    config_defaults = {"cell_type": "lstm",
                       "num_layers": 4,
                       "hidden_size": 256,
                       "num_epochs":10,
                       "bi_dir":True,
                       "dropout": 0.2,
                       "embed_size":256
                      } 

    wandb.init(config=config_defaults, project=project_name, resume=False)
    
    config = wandb.config 
    
    
    learning_rate = 0.001
    load_model = False
    num_epochs = config.num_epochs
    encoder_embedding_size = config.embed_size
    decoder_embedding_size = config.embed_size
    input_size_encoder = len(train_data.vocab_eng)
    input_size_decoder = len(train_data.vocab_hin)
    output_size = len(train_data.vocab_hin)
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    enc_dropout = config.dropout
    dec_dropout = config.dropout
    cell_type = config.cell_type
    bi_directional = config.bi_dir
    Test = True

    wandb.run.name  = "test_{}_cell_{}_nl_{}_hs_{}_e_{}_bd_{}_dr_{}_ems_{}".format(Test, cell_type,
                                                                          num_layers,
                                                                          hidden_size,
                                                                          num_epochs,
                                                                          bi_directional,
                                                                          enc_dropout,
                                                                          encoder_embedding_size
                                                                          )
                                                                              
                                                                                  
  
    print(wandb.run.name )
    
    encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                         hidden_size, num_layers, enc_dropout, cell_type,
                          bi_directional).to(device)

    decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                         hidden_size, output_size, num_layers, dec_dropout, 
                          cell_type ,bi_directional).to(device)

    model = Seq2Seq(encoder_net, decoder_net, cell_type).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    pad_index = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

    if load_model:
        load_checkpoint(torch.load('my_checkpoint.pth.ptar'),model, optimizer)
    
    print("Training the model.....")
    for epoch in range(num_epochs):
        print('Epoch: ', epoch+1, '/', num_epochs)

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            input_word = inputs.to(device)
            target_word = targets.to(device)

            output = model(input_word, target_word)
            # output shape: (target_len, batch_size, output_vocab_size)

            output = output[1:].reshape(-1, output.shape[2])
            target_word = target_word[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target_word)

            loss.backward()

            # To handle large gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        print("Training Loss: ", loss.item())
        
        train_loss = loss.item()
        if (epoch+1)%2 == 0:
            model.eval()
            print("Finding accuracy of the model.....")
            train_accu =  accuracy(train_data, model, cell_type)
            valid_accu =  accuracy(valid_data, model, cell_type)
            model.train()

            wandb.log({"valid accuracy": valid_accu, "train accuracy": train_accu,
                        "train loss": train_loss, 'epoch': epoch+1})
        else:
            wandb.log({"train loss": train_loss, 'epoch': epoch+1}) 
            
    ## For testing the model uncomment the code
#     model.eval()        
#     test_accur =  accuracy(test_data, model, cell_type) 

     # Uncomment  prediction_csv only if you want to generate the csv file of predictions
#     prediction_csv(test_data, model, cell_type)  
#     model.train()                                            
    
#     wandb.log({"test accuracy": test_accur})
        
    wandb.run.finish()
    


# In[18]:


#train_with_wandb()


# In[22]:


hyperparameters = {

        "num_layers": {
            "values": [2, 3, 4]
        },
        "hidden_size": {
            "values": [64, 128, 256]
        },
        "cell_type": {
            "values": ["rnn", "gru", "lstm"]
        },
        "num_epochs":{
            "values": [10, 15, 20]
        },
        "bi_dir":{
            "values": [False, True]
        },
        "dropout": {
            "values": [0.2, 0.3, 0.5]
        },
        "embed_size":{
            "values": [64, 128, 256]
        },
  }


# In[23]:


def wandb_sweep(project_name, entity_name):
    '''
    This function is used to run the wandb sweeps. 
    It takes in project name and entity name as input , and does not return any value.

    '''
    sweep_config={

      "method": "bayes",
      "metric": {
          "name": "valid_accu", 
          "goal": "maximize"
          },
      "parameters":hyperparameters
    }

    sweep_id=wandb.sweep(sweep_config, project=project_name, entity=entity_name)
    wandb.agent(sweep_id,train_with_wandb)


# In[23]:


if wandb_sweep == True:
    wandb_sweep(project_name, entity_name)

