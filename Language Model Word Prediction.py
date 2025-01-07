import nltk
from nltk.corpus import brown, stopwords
from collections import Counter

nltk.download('brown')
nltk.download('stopwords')

def list_bigram(word, corpus_words):
    # Finding all bigrams starting with the word entered 
    bigrams = []
    for i in range(len(corpus_words)-1):
        if corpus_words[i].lower() == word:
            bigrams.append((word, corpus_words[i+1]))
    
    # Counting frequencies of bigrams
    bigram_counts = Counter(bigrams)
    word_count = sum(count for (w1), count in bigram_counts.items() if w1 == word)
    
    # Calculating probabilities for each next word
    probability = {}
    for (w1, w2), count in bigram_counts.items():
        if w1 == word:
            probability[w2] = count / word_count
    
    # Getting top 3 most likely words 
    return sorted(probability.items(), key=lambda x: x[1], reverse=True)[:3]

def main():
    # Getting list of stopwords 
    stop_words = set(stopwords.words('english'))
    
    # Processing corpus keeping original form but removing stopwords
    corpus_words = []
    for word in brown.words():
        if word.lower() not in stop_words:
            corpus_words.append(word)

    sent = []
    while True:
     
        if not sent:
            while True:
                word = input("\nEnter a word to start with: ").lower()
                if word in [w.lower() for w in corpus_words]:
                    sent.append(word)
                    break
                print(f"'{word}' not found in corpus.")
                menu = input("Enter 1 to try again, or 2 to quit: ")
                if menu == '2':
                    return
        
        # Getting next word options
        current_word = sent[-1]
        next_words = list_bigram(current_word, corpus_words)
        
        if not next_words:
            print("\nNo next words found")
            break
            
        # Display menu
        print(f"\nCurrent sentence: {' '.join(sent)} ...")
        print("\nWhich word should follow:")
        for i, (word, prob) in enumerate(next_words, 1):
            print(f"{i}) {word} P({current_word} {word}) = {prob:.3f}")
        print("4) QUIT")
        
        # Giving user choice
        menu = input("\nEnter your choice (1,2,3,4): ")
        if menu == '4':
            break
        elif menu not in ['1', '2', '3']:
            menu = '1'  # Setting 1 as default to first option
            
        sent.append(next_words[int(menu)-1][0])
    
    print(f"\nFinal sentence: {' '.join(sent)}")

if __name__ == "__main__":
    main()