from sklearn.neighbors import KNeighborsClassifier
import nltk
from nltk.corpus import brown 
from nltk.corpus import cess_esp
from nltk.corpus import wordnet

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            text = brown.words()

        else:  # spanish
            self.avg_word_length = 6.2
            text = cess_esp.words()


        self.fdist = nltk.FreqDist(w.lower() for w in text)
        self.model = KNeighborsClassifier()
        self.total = len(text)


    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        freq = self.fdist[word]
        lemmas = []
        hypernyms = []
        hyponyms = []

        #number of senses 
        if self.language == 'english':
            syn = wordnet.synsets(word)
        else:
            syn = wordnet.synsets(word, lang='spa')
        syn_no = len(syn)

        #number of synonyms, hypernyms, hyponyms  
        for s in syn:
            hypernym_synset = s.hypernyms()
            for synset in hypernym_synset:
                for lemma in synset.lemma_names():
                    hypernyms.append(lemma)
            hyponym_synset = s.hyponyms()
            for synset in hyponym_synset:
                for lemma in synset.lemma_names():
                    hyponyms.append(lemma)
            for l in s.lemma_names():
                lemmas.append(l)
        lemma_no = len(lemmas)
        hypernym_no = len(hypernyms)
        hyponym_no = len(hyponyms)

        lemma_freq = 0 
        for lemma in lemmas:
            lemma_freq += self.fdist[lemma]

        if lemma_no > 0:
            avg_lemma_freq = lemma_freq/lemma_no
        else:
            avg_lemma_freq = 0 
            
        if freq > avg_lemma_freq:
            syn_freq = 1
        else:
            syn_freq = 0 

        return [len_chars, len_tokens, freq, syn_no, lemma_no, hyponym_no, hypernym_no, syn_freq]

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])

        self.model.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))

        return self.model.predict(X)
