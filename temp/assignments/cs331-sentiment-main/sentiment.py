# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions
import re
import classifier 


def process_text(filename):
    """
    Preprocesses the text: Remove apostrophes, punctuation marks, etc.
    Returns a list of text
    """
    file_lines = []
    #nltk.download('stopwords')
    #from nltk.corpus import stopwords
    #stop_words = set(stopwords.words('english'))
    with open(f'{filename}', 'rt') as file:
        for line in file:
            file_lines.append(line.rstrip('\n'))
    
    for x, line in enumerate(file_lines):

        line = re.sub('','', line)    
        line = re.sub('https://,*', ' ',line)
        line = line.translate({ord(i): None for i in '.,?!\'\":;()[]-_'})
        line = line.translate({ord(i): ' ' for i in '\\/'})
        line = line.replace('  ', ' ')
        line = line.lower()
        line = re.split(' ',  line)
        line.pop(-1)
        line.pop(-2)
        file_lines[x] = line

    return file_lines


def build_vocab(preprocessed_text, list=None):
    """
    Builds the vocab from the preprocessed text
    preprocessed_text: output from process_text
    Returns unique text tokens
    """
    present = False
    
    if list == None:
        vocab = []
    else:
        vocab = list
    
    for x, line in enumerate(preprocessed_text):
        for elem in line[:-1]:
            if vocab == []:
                vocab.append(elem)
            for word in vocab:
                if elem == word:
                    present = True
                    break
            if present != True:
                vocab.append(elem)
            else:
                present = False
    
    vocab.sort()  
    return vocab


def vectorize_text(text, vocab):
    """
    Converts the text into vectors
    text: preprocess_text from process_text
    vocab: vocab from build_vocab
    Returns the vectorized text and the labels
    """
    word_to_index = {word: index for index, word in enumerate(vocab)}
    
    vectors = []
    labels = []
    for line in text:
        vector = [0] * len(vocab)
        # the sentence format is ['token1', 'token2', ..., 'classlabel']
        for word in line[:-1]:  
            if word in word_to_index:
                vector[word_to_index[word]] = 1
        vectors.append(vector)
        labels.append(line[-1])
        
    return vectors, labels

def create_preprocessed_files(vectors, vocab, labels, filename):
    
    with open(filename, 'w') as f:
        # print feature names
        f.write(','.join(vocab) + ',classlabel\n')

        # print each bag of words followed by the corresponding label
        for i in range(len(vectors)):
            f.write(','.join(map(str, vectors[i])) + ',' + str(labels[i]) + '\n')    

def accuracy(dataset, predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """
    assert len(predicted_labels) == len(true_labels), "The lists of predicted and true labels must have the same length"

    correct_predictions = sum(p == t for p, t in zip(predicted_labels, true_labels))
    total_predictions = len(predicted_labels)

    accuracy = correct_predictions / total_predictions
    print(f"{dataset}" + " Accuracy: {:.2f}%".format(accuracy * 100))


    return accuracy


def main():
    # Take in text files and outputs sentiment scores
    pp_train = process_text('trainingSet.txt')
    pp_test = process_text('testSet.txt')
    
    vocab = build_vocab(pp_train)
    #vocab = build_vocab(pp_test, list=vocab)
    
    v_train, v_train_labels = vectorize_text(pp_train, vocab)
    v_test, v_test_labels = vectorize_text(pp_test, vocab)
    
    create_preprocessed_files(v_train, vocab, v_train_labels, 'preprocessed_train.txt')
    create_preprocessed_files(v_test, vocab, v_test_labels, 'preprocessed_test.txt')
    
    
    model = classifier.BayesClassifier(vocab)
    model.train(pp_train)

    train_predicted_labels = model.classify_text(pp_train)
    test_predicted_labels = model.classify_text(pp_test)
    #print(predicted_labels)
    #print(v_train_labels)
    accuracy('Train', train_predicted_labels, v_train_labels)
    accuracy('Test', test_predicted_labels, v_test_labels)

if __name__ == "__main__":
    main()