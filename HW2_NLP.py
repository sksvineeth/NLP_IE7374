#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import BlanklineTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence


# In[19]:


#Read File Content

file_content = open("NLPdataHW2.txt").read()


# In[20]:


#1) Tokenize the text using word tokenize. Convert these tokens to lowercase. 
tokens = nltk.word_tokenize(file_content)
print(tokens)
tokenized_text = [word.lower() for word in tokens]
print(tokenized_text)
print("Length", len(tokenized_text))
# Also, calculate the number of tokens and determine the frequency of each word. List ten most-frequent tokens.
data_analysis = nltk.FreqDist(tokenized_text)
data_analysis.most_common(10)


# In[ ]:


#2)Tokenize the text using NLTK blankline tokenizer. Your output should include the list of tokens.

print("Given file without linespaces" , BlanklineTokenizer().tokenize(file_content))
file_content_withlinespaces = open("NLPdataHW22.txt").read()
BlanklineTokenizer().tokenize(file_content_withlinespaces) # we can see the delimiter linespace here


# In[ ]:


#3)Tokenize the text using Tensorflow Keras API. 
#Follow the instructions provided in the starter code regarding the use of various functions for this question.

# Tokenize the file_content

# Initializing Function Parameters
num_words = 1000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'

# Tokenize 
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
print(tokenizer.fit_on_texts(file_content))

# tokenize the document
result = text_to_word_sequence(file_content)
print(result)

#Also determine the indices of the generated word tokens.
# Get file_content word index
word_index = tokenizer.word_index
word_index




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


tokens


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


file_content


# In[ ]:


BlanklineTokenizer().tokenize(file_content)


# In[ ]:


num_words = 1000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



# tokenize the document
result = text_to_word_sequence(file_content)
print(result)


# Ngrams, in this case bi-gram (n = 2)

bigrams = result.ngrams(tokens, 2, reduction_type=text.Reduction.STRING_JOIN)

print(bigrams.to_list())


# In[ ]:




