# CS6910 Assignment -3
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
Created Encoder, Decoder which will be used for Seq2Seq model creation.

### 7. Function for training the model:
- Created __train()__ function to train our model.
- In this, first model is defined and it is exported to the device (either GPU or CPU).
- Optimizer and loss function is imported using __torch__ library.
  __optimizer__=torch.optim.__Adam__(model.parameters(),__lr__=0.0001,__weight_decay__=0.0001)
  __loss_function__=nn.__CrossEntropyLoss__()
- __train_with_wandb__ is the second function to train our model if we want to run sweep using wandb.
  I have also include the commands needed to integrate the __wandb__ __sweep__. I have __login__ to wandb account. I have already imported the __wandb__, now I am giving the __default values__ of our 
  variable for __sweep__. After that I have defined the __wandb run name__ which will be assign to each run. Values like __epoch__, __train loss__, __train accuracy__ and __validation accuracy__ are
  login  to wandb.
- __Saving__ the wandb run and __finishing__ the run

### 9. Arg parse 

created function __arg_parse()__ to pass the command line arguments.

- Using argparse, I have define the arguments and options that my program accepts,
- argparse will run the code, pass arguments from command line and 
    automatically generate help messages.
- __I have given the defaults values for 
    all the arguments, so code can be run without passing any arguments.__
    
Description of various command line arguments

    --wandb_sweep : Do you want to sweep or not: Enter True or False. Default value is False. 
    --wandb_entity : Login username for wandb. Default is given but if you are already using wandb, you will be logged in automatically.
    --wandb_project : name to initialize your run. No need to mention if you are just trying the code.
    --data_augmentation : Data Augmentation: True or False
    --epochs : Number of Epochs: integer value
    --batch_size : Batch Size: integer value
    --dense_layer : Dense Layer size: integer value
    --activation : Activation function: string value
    --batch_normalisation : Batch Normalization in each layer: True or False
    
### 10. Training our CNN model
I have created training file called __train_partA.py__ file it has everything needed for training and testing our model. we can run the code using command line arguments. 

Or we may also use .ipynb file for partA problem to train the model and test it.

### 11. Running the wandb sweep:

The wandb configuration for sweep:
- sweep_config = {"name": "cs6910_assignment2", "method": "bayes"}   
- sweep_config["metric"] = {"name": "val_accuracy", "goal": "maximize"}

- parameters_dict = {
            
             "num_filters": {"values": [[12,12,12,12,12],[4,8,16,32,64],[64,32,16,8,4]},
              "act_fu": {"values": ["relu","selu","mish"]},
              "size_kernel": {"values": [[(3,3),(3,3),(3,3),(3,3),(3,3)], [(3,3),(5,5),(5,5),(7,7),(7,7)],
                                         [(7,7),(7,7),(5,5),(5,5),(3,3)]]}, 
                "data_augmentation": {"values": [True, False]} ,
                "batch_normalisation": {"values": [True, False]} ,
                "dropout_rate": {"values": [0, 0.2, 0.3]},
                "size_denseLayer": {"values": [50, 100, 150, 200]}
                }
- sweep_config["parameters"] = parameters_dict

- sweep_id = wandb.sweep(sweep_config, entity="am22s020", project="cs6910_assignment2")
- wandb.agent(sweep_id, train_CNN, count=150)


