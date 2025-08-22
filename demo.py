import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize
from nltk import word_tokenize, pos_tag, ne_chunk
import os


print("NLTK version:", nltk.__version__)
nltk.download('all')

# tokenize a sample text  
# word tokenization and sentence tokenization
sample_text = "Hello, world! This is a sample text for tokenization."
tokens = nltk.word_tokenize(sample_text)
print("Tokens:", tokens)

# sentence tokenization
sent = "GeeksforGeeks is a great learning platform.\
It is one of the best for Computer Science students."
print(word_tokenize(sent))
print(sent_tokenize(sent))


# stemming and lemmatization
from nltk.stem import PorterStemmer

# create an object of class PorterStemmer
porter = PorterStemmer()
print(porter.stem("play"))
print(porter.stem("playing"))
print(porter.stem("plays"))
print(porter.stem("played"))


# lemmatization
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("plays", 'v'))
print(lemmatizer.lemmatize("played", 'v'))
print(lemmatizer.lemmatize("play", 'v'))
print(lemmatizer.lemmatize("playing", 'v'))


# part of speech tagging
text = "GeeksforGeeks is a Computer Science platform."
tokenized_text = word_tokenize(text)
tags = tokens_tag = pos_tag(tokenized_text)
print(tags)

# Named Entity Recognition (NER)
# Download the required resource for NER
nltk.download('maxent_ne_chunker_tab')
nltk.download('words') # This resource is also needed for the chunker

# Sample text
text = "Barack Obama was born in Hawaii in 1961."

# Tokenize and POS tag the sentence
tokens = word_tokenize(text)
tags = pos_tag(tokens)

# Apply Named Entity Recognition
entities = ne_chunk(tags)
print(entities)

print(os.listdir(nltk.data.find('corpora')))