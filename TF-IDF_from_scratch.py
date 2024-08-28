import numpy as np
from nltk.tokenize import word_tokenize
import nltk

# Assign two sample documents
d0 = "The cat sat on the wall"
d1 = "The dog sat on the mat"

# Combine the documents into a sample text list
sample_text = [d0, d1]
print(f"sample text : \n {sample_text}")

# Initialize empty lists for sentences and word set
sentences = []
word_set = []

# Tokenize each sentence and build the word set
for sent in sample_text:
    # Convert each word to lowercase and filter out non-alphabetic tokens
    words = [word.lower() for word in word_tokenize(sent) if word.isalpha()]
    sentences.append(words)
    # Add unique words to the word set
    for word in words:
        if word not in word_set:
            word_set.append(word)

# Convert word_set to a set of unique words
word_set = set(word_set)
print(f"word_set :{word_set}")

# print the length, total_documents, total words
total_docs = len(sample_text)
print('Total documents:', total_docs)
print('Total words:', len(word_set))

# Function to count the occurrences of each word in the entire corpus
def count_dict(sentences):
    count_dict = {}
    # Initialize the count dictionary with each word set to 0
    for word in word_set:
        count_dict[word] = 0
    # Count occurrences of each word in the sentences
    for sent in sentences:
        for word in sent:
            count_dict[word] += 1
    return count_dict

# Get the word count for the entire corpus
word_count = count_dict(sentences)
print('Word Count:', word_count)

# Function to calculate term frequency (TF) of a word in a document
def term_frequency(document, word):
    N = len(document)  # Total number of words in the document
    occurance = len([token for token in document if token == word])  # Count occurrences of the word
    return occurance / N  # TF is the ratio of the word occurrences to the total words

# Function to calculate inverse document frequency (IDF) of a word
def inverse_document_frequency(word):
    try:
        word_occurance = word_count[word] + 1  # Add 1 to avoid division by zero
    except:
        word_occurance = 1
    return np.log(total_docs / word_occurance)  # IDF is the log of total docs divided by word occurrences

# Function to create the TF-IDF vector for a sentence
def tf_idf(sentence):
    vec = np.zeros((len(word_set),))  # Initialize a zero vector of length equal to the number of unique words
    for word in sentence:
        tf = term_frequency(sentence, word)  # Calculate term frequency for the word
        idf = inverse_document_frequency(word)  # Calculate inverse document frequency for the word
        word_idx = list(word_set).index(word)  # Get the index of the word in the word set
        vec[word_idx] = tf * idf  # Set the TF-IDF value in the vector
    return vec

# Generate TF-IDF vectors for all sentences
vectors = []
for sent in sentences:
    vectors.append(tf_idf(sent))  # Append the TF-IDF vector of the sentence to the list

print("TF-IDF Vectors:")
print(vectors)  # Print the TF-IDF vectors
