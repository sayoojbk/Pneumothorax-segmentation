#### The flow of the model
- Firstly what each files and folder are for.
    1) siimdata:
       Cotains train,test,train_rle.csv : train,test have not beeen pushed.
       LINK : https://www.kaggle.com/abhishek/siim-png-images
       
    2) data_loader_rle.py :
      This file creates the data input pipeline to be fed to the model network. The main methods are the _len__() 
      and the __getitem__()[__getitem__() is what is makes it possible to call element of dataset.]
      THe torch.dataloader creates a Pytorch dataloader to be called. Works as a generator calling the data in
      size of the batch size.
      
    3) Unet.sh  : Bash code to run the model.
    4) evalutation.py : The evaluation code for the model.Haven't run it yet. I will need  to make changes here 
       will do once I feel the model is training properly i.e the loss is decreasing.
    5) Solver   : The Class defining the model and the training code.   
    6) main.py  : This is the script which runs the training of the model by calling the SOlver Class.
    7) Network  : Basic backbone model in Pytorch containing the diff models I am using .
    8) debug.py : This is just a useless file. Used it to check if my dataloader was working properly or not.
                  Contains some code snippets to just check if the dataloader or model are building properly.
                  
    9) utils.py : Contains the rle2mask and mask2rle codes. Nothing some utility codes to be added here.
    
