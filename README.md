# RNN based transliteration model
Used recurrent neural networks to build a transliteration system.

## Authors

 [@Haider Altaf am22s020](https://www.github.com/HaiderAltaf)

## General Instructions:

Install the required libraries using the following command :

    pip install -r requirements.txt

Along with jupyter notebooks we have also given python code filed (.py) files. These contain the code to direclty train and test the CNN in a non interactive way.

If you are running the jupyter notebooks on colab, the libraries from the requirements.txt file are preinstalled, with the exception of wandb. You can install wandb by using the following command :

    !pip install wandb

The dataset for this project can be found at :

    https://ai4bharat.org/aksharantar
    
### 1. References:

-     https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

-     https://towardsdatascience.com/what-is-teacher-forcing-3da6217fed1c

-     https://www.youtube.com/watch?v=9sHcLvVXsns&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=10


-     https://www.youtube.com/watch?v=EoGUlvhRYpk&t=2169s

 
-     https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/Seq2Seq/seq2seq.py

      
### 2. Libraries Used: 

- Used the pytoch, __pytorchvision__ and other necessary libraries for the CNN model buiding and training.

### 3. Device:
I have write and run the code in Jupyter notebook and used GPU of my system for training.

### 4. Data Pre-processing:

- Created a function named __Vocabulary__ to builds a character-level vocabulary for a given list of words.
- __aksharantar__ function created to process text data for a machine translation task.
- __MyCollate__ function used to collate the data items into batches for DataLoader.
- __get_loader__ function created to give a PyTorch DataLoader object and a custom dataset object.

### 5. Installing wandb and importing wandb:
Run the below code if you want to run the sweep
- !pip install wandb
- import wandb

### 6. RNN Model :
- Created Encoder, Decoder which will be used for Seq2Seq model creation.

### 7. Function for training the model:
- Created __train()__ function to train our model.
- __train_with_wandb__ is the second function to train our model if we want to run sweep using wandb.
- These functions are same in the case of model with attention and without attention

### 9. Arg parse 
To pass the command line arguments.
- argparse will run the code, pass arguments from command line and 
    automatically generate help messages.
- __I have given the defaults values for 
    all the arguments, so code can be run without passing any arguments.__
    
Description of various command line arguments

    --wandb_sweep : Do you want to sweep or not: Enter True or False. Default value is False. 
    --wandb_entity : Login username for wandb. Default is given but if you are already using wandb, you will be logged in automatically.
    --wandb_project : name to initialize your run. No need to mention if you are just trying the code.
    --cell_type : RNN cell types: 'lstm', 'rnn', or 'gru'
    --epochs : Number of Epochs: integer value
    --hidden_size : number of units or neurons in the hidden layer of the network : integer value
    --embedding_size :  The embedding size is the dimensionality of the dense vector representation: integer value
    --num_layers : number of recurrent layers that are stacked on top of each other to process sequential input: string value
    --bi_directional : input sequence to be processed in both forward and backward directions: True or False
    
### 10. Training our CNN model
I have created training file called __Seq2Seq_vanilla.py__ and  __Seq2Seq_attention.py__ files it has everything needed for training and testing our model
for the model without attention and with atention respectively.
we can run the code using command line arguments. 

Or we may also use __.ipynb files__ for both the part to train the model and test it by manually running the code.

### 11. Running the wandb sweep:

The wandb configuration for sweep:
- sweep_config = {"name": "cs6910_assignment 3", "method": "bayes"}   
- sweep_config["metric"] = {"name": "val_accuracy", "goal": "maximize"}

- hyperparameters = {
        "num_layers": { "values": [2, 3, 4] },
        "hidden_size": {  "values": [64, 128, 256]},
        "cell_type": { "values": ["rnn", "gru", "lstm"]},
        "num_epochs":{  "values": [10, 15, 20]},
        "bi_dir":{   "values": [False, True]},
        "dropout": { "values": [0.2, 0.3, 0.5] },
        "embed_size":{ "values": [64, 128, 256]},
      }

- sweep_config["parameters"] = hyperparameters

- sweep_id = wandb.sweep(sweep_config, entity="am22s020", project="cs6910_assignment2")

To run the wandb sweep, you need to give the wandb_sweep=True through command line arguments or
for manual you may run the code named __train_with_wandb__.

### 12. Evaluating the model:

 To find the validation and train accuracy we may run the __train()__ function by providing the values 
 of hyperparameters of the model. 
 
 I have written the function to predict the hindi word of the input word then function created to find the accuracy.
 
 For testing the model, just change the dataset argument of the __accuracy__ function as test data to find the 
 test accuracy at the given hyper parameters values in the __train__ function.

