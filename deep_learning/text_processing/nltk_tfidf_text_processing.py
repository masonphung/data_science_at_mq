import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import pandas as pd

# Task 1
def topN_pos(csv_file_path, N):
    """
    Reads the 'train.csv', exact the unique questions from `qtext`.
    It tokenizes each sentence and word, tag them universally.
    The function then returns top N most common nouns.
    
    Parameters:
    csv_file_path (str): Directory of the data
    N (int): number of words returned
    
    Returns:
    Top N word with the most occurence count.
    
    Note: Data should have a `qtext` variable.
    Example:
    >>> topN_pos('train.csv', 3)
    output would look like [(noun1, 22), (noun2, 10), ...]
    """
    # Load the data based on the path
    df = pd.read_csv(csv_file_path)
    # Get unique observations of `qtext`
    qtext = df['qtext'].unique()
    # Join the array elements together
    joined_sents = ' '.join(str(sent) for sent in qtext)
    
    # Tokenize the data then tag each word
    tokenized_sents = nltk.sent_tokenize(joined_sents)
    tokenized_words = [nltk.word_tokenize(sent) for sent in tokenized_sents]
    tagged = nltk.pos_tag_sents(tokenized_words, tagset = 'universal')
    
    # Get only nouns
    noun_list = []
    for sent in tagged:
        for word in sent:
            if word[1] == 'NOUN':
                noun_list.append(word[0])
    
    # Count the words in the noun_list
    count = collections.Counter(noun_list)
    # Return the most common N words
    return count.most_common(N).dtypes     
    


# Task 2
def topN_2grams(csv_file_path, N):
    """
    Reads the 'train.csv', exact the unique questions from `qtext` and tokenizes each sentence and word.
    For both stem and non-stem bigrams:
        - In each sentence:
            - Loop through each word in a sentence, stem (if needed), 
            then append to a temporary stem and non-stem list (the lists only for the sentence).
            - Determine the bigrams though the temporary lists.
            - Append the bigrams into the general/big stem and non-stem bigram lists.
        - Then determine the frequency distribution and normalize them to get the result.
        
    Parameters:
    csv_file_path (str): Directory of the data
    N (int): number of bigrams returned
    
    Returns:
    Top N bigrams with the most occurence count for stem and non-stem, respectively.
    
    Note: Data should have a `qtext` variable.
    Example:
    >>> topN_2grams('train.csv', 3)
    output would look like [('what', 'is', 0.4113), ('how', 'many', 0.2139), ....], [('I', 'feel', 0.1264), ('pain', 'in', 0.2132), ...]
    """
    # Load the data based on the path
    df = pd.read_csv(csv_file_path)
    # Get unique observations of `qtext`
    qtext = df['qtext'].unique()
    # Join the array elements together
    joined_sents = ' '.join(str(sent) for sent in qtext)
    # Tokenize the data then tag each word
    tokenized_sents = nltk.sent_tokenize(joined_sents)
    tokenized_words = [nltk.word_tokenize(sent) for sent in tokenized_sents]
    
    # Create bigram lists for both cases
    stem_bigrams_list = []
    non_stem_bigrams_list = []
    # Create a PorterStemmer object
    stemmer = nltk.PorterStemmer()
    
    # Run a loop through each sentence and each word in the sentence
    # We will do the work for both stem and non-stem in each step
    for sent in tokenized_words:
        # Create a temp list that refreshes by each sentence
        stemmed_list = []
        non_stem_list = []
        for word in sent:
            # For stem
            stemmed = stemmer.stem(word)
            stemmed_list.append(stemmed)
            # Non-stem
            non_stem_list.append(word)
        # Determine bigrams in the sentence
        stem_bigrams = list(nltk.bigrams(stemmed_list))
        non_stem_bigrams = list(nltk.bigrams(non_stem_list))
        # Add each bigram from the temp list to the final list
        for each_stem in stem_bigrams:
            stem_bigrams_list.append(each_stem)
        for each_non_stem in non_stem_bigrams:
            non_stem_bigrams_list.append(each_non_stem)
         
    # Determine the frequency distribution of both list
    stem_freq_dist = nltk.FreqDist(stem_bigrams_list)
    non_stem_freq_dist = nltk.FreqDist(non_stem_bigrams_list)
    # Determine the total number of bigrams
    total_stem = stem_freq_dist.N()
    total_non_stem = non_stem_freq_dist.N()
    # Loop through each word in the list to get the normalized frequency
    for word in stem_freq_dist:
        stem_freq_dist[word] = round(stem_freq_dist[word] / float(total_stem), 4) # Round to 4
    for word in non_stem_freq_dist:
        non_stem_freq_dist[word] = round(non_stem_freq_dist[word] / float(total_non_stem), 4) # Round to 4
    
    return (stem_freq_dist.most_common(N), non_stem_freq_dist.most_common(N))
        
        
        
# Task 3
def sim_tfidf(csv_file_path):
    """
    Reads the 'train.csv' and exact the unique data of `qtext` and 'atext' then fit the them into a Tfidf vectorizer.
    Then, loop through each question:
        - Get its corresponding answers and transform both questions and answers into array.
        - Do the matrix multiplication to find the cosine similarity, get the answer with the highest similarity 
        and record if its label = 1.
    Calculate the proportion of correct answers by dividing the total of correct answers with the total number of answers.
        
    Parameters:
    csv_file_path (str): Directory of the data
    
    Returns:
    The proportion of correct answers, rounded by 2.
    
    Data should have 'qtext', 'atext' and 'label' variables
    Example:
    >>> sim_tfidf('train.csv')
    output format would be like 0.54
    """
    # Load data, then take the unique observation of 'qtext' and 'atext'
    df = pd.read_csv(csv_file_path)
    q_unique = df['qtext'].unique()
    a_unique = df['atext'].unique()
    train_unique = np.concatenate([q_unique, a_unique]) # Join them together
    
    # Create a TfidfVectorizer object then fit the data
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(train_unique)
    
    # Create a correct answer counter
    correct_answers = 0
    # Loop through each unique question
    for each in q_unique:
        # Only take the data of the question
        data = df[df['qtext'] == each]
        # Transform the question and its answers
        q_tfidf = tfidf.transform([each]).toarray()
        a_tfidf = tfidf.transform(data['atext']).toarray()
        
        # Compute cosine similarity between each question and its corresponding answers
        # Let's transpose the matrix as we have different vector lengths with `.T`
        cosine_sim = np.dot(q_tfidf, a_tfidf.T)
        # Get the answer with the highest similarity
        max_sim_idx = cosine_sim.argmax()

        # Check if that answer of has a label of 1
        # Make sure to re-filter the data based on the chosen question.
        if data['label'][data.index[max_sim_idx]] == 1:
            # If yes, increase the total of correct answers by one
            correct_answers += 1
    
    # Find the proportion of the correct answers = correct ans / total ans
    proportion_correct = correct_answers / len(q_unique)
    # Return the prop and round by 2 as required
    return round(proportion_correct, 2) 

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    print(topN_pos('train.csv', 3))
    print("------------------------")
    print(topN_2grams('train.csv', 3))
    print("------------------------")
    print(sim_tfidf('train.csv'))

