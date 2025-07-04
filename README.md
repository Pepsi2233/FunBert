# FunBert
A method for classifying fungi

# Usage steps
1、Use phylum.py/class.py/order.py/family.py/genus.py to categorise .data on the phylum hierarchy to get a folder A containing all phylum/Class/order/family/genus categories.  
2、Use 0.More_Than_n_File.py to count the list of classes with at least n reference sequences in the phylum/Class/order/family/genus category B.  
3、Use 0.Funtine_data_process.py input list B to generate training, validation and test sets on phylum/Class/order/family/genus classes.  
4、Training the model using the train.py file.  
5、Testing the model using the Test.py file.  
