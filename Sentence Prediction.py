# Importing modules 
import nltk
from nltk.corpus import brown
from collections import defaultdict

def download_browncorpus(): 
    try:
        nltk.download('brown', quiet=True)
        return True
    except:
        print("Error downloading")
        return False

def main():
    if not download_browncorpus():
        return

    S = input("Enter a sentence: ")
    S = S.lower()  # Convert to lowercase
    
    words = S.split() # Get input sentence from user
    
    # Checking sentence length
    if len(words) < 2:
        print("Error: Please enter a sentence with at least two words.")
        return
    
    print(f"\nCalculating bigrams: {S}")
    
    # Initializing counters
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)    

    # Counting frequencies directly from Brown corpus
    for sentence_S in brown.sents():
        sentence_S = [word.lower() for word in sentence_S] # Convert words to lowercase

        # Counting unigrams 
        for word in sentence_S: 
            unigram_counts[word] += 1
        
        # Counting bigrams
        for i in range(len(sentence_S)-1):
            bigram = (sentence_S[i], sentence_S[i+1])
            bigram_counts[bigram] += 1
    
    # Creating a list of bigrams from entered sentence
    S_bigrams = []
    for i in range(len(words)-1):
        S_bigrams.append((words[i], words[i+1]))
    
    # Calculate total probability
    SE_probability = 0.25  # Starting probability
    
    print("\nProbability Calculations:")
    print("-" * 40)
    print(f"Starting probability = 0.25")
    
    # Calculate and display each bigram probability
    print("\nBigram probabilities:")
    for bigram in S_bigrams:
        w1, w2 = bigram
        # Calculating using the formula  P(word2|word1) = count(word1,word2)/count(word1)
        if unigram_counts[w1] > 0:
            Bigram_probability = bigram_counts[bigram] / unigram_counts[w1]
        else:
            Bigram_probability = 0
        print(f"P({w2}|{w1}) = {Bigram_probability:.6f}")
        SE_probability *= Bigram_probability
    
    # Multiply by end probability
    SE_probability *= 0.25
    print(f"End probability = 0.25")
    
    # Displaying results
    print("\nResults:")
    print("-" * 40)
    print(f"Entered sentence: {sentence_S}")
    print("\nBigrams in sentence:")
    for bigram in S_bigrams:
        print(f"({bigram[0]}, {bigram[1]})")
    
    #probability calculation for the sentence 
    print("\nFinal probability :")
    print(f"P(\"{sentence_S}\") = 0.25 * ", end='')
    for i, bigram in enumerate(S_bigrams):
        if i < len(S_bigrams) - 1:
            print(f'P({bigram[1]}|{bigram[0]}) * ', end='')
        else:
            print(f'P({bigram[1]}|{bigram[0]}) * 0.25')
    
    print(f"\nFinal probability: {SE_probability:.10f}")

if __name__ == '__main__':
    main()
