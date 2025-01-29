# This file implements a Naive Bayes Classifier
import math

class BayesClassifier:
    """
    Naive Bayes Classifier
    file length: file length of training file
    sections: sections for incremental training
    """
    def __init__(self, vocab):
        self.vocab = vocab
        self.positive_word_counts = {word: 1 for word in vocab}  # Uniform Dirichlet prior
        self.negative_word_counts = {word: 1 for word in vocab}  # Uniform Dirichlet prior
        self.positive_sentence_counts = 0
        self.negative_sentence_counts = 0

    def train(self, train_data):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab
        """
        for sentence in train_data:
            #print(sentence)
            if sentence[-1] == '1':
                #print('here')
                self.positive_sentence_counts += 1
                for word in sentence[:-1]:
                    #print(word)
                    self.positive_word_counts[word] += 1
            else:
                
                self.negative_sentence_counts += 1
                for word in sentence[:-1]:
                    self.negative_word_counts[word] += 1

    def classify_text(self, test_data):
        """
        vectors: [vector1, vector2, ...]
        predictions: [0, 1, ...]
        """
        total_sentences = self.positive_sentence_counts + self.negative_sentence_counts
        log_prob_positive = math.log(self.positive_sentence_counts / total_sentences)
        log_prob_negative = math.log(self.negative_sentence_counts / total_sentences)

        predictions = []

        for sentence in test_data:
            positive_score = log_prob_positive
            negative_score = log_prob_negative

            for word in self.vocab:
                #print(f'Sentence: {sentence}\nWord: {word}')
                if word in sentence[:-1]:
                    positive_score += math.log(self.positive_word_counts[word] / self.positive_sentence_counts)
                    negative_score += math.log(self.negative_word_counts[word] / self.negative_sentence_counts)
                else:
                    positive_score += math.log(1.0 - (self.positive_word_counts[word] / self.positive_sentence_counts))
                    negative_score += math.log(1.0 - (self.negative_word_counts[word] / self.negative_sentence_counts))
            #print(f'Positive: {positive_score} Negative: {negative_score}')
            if positive_score > negative_score:
                predictions.append('1')
            else:
                predictions.append('0')
        return predictions

    