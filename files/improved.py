from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import nltk
from nltk.corpus import brown 
from nltk.corpus import cess_esp
from nltk.corpus import wordnet

class Improved(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            text = brown.words()

        else:  # spanish
            self.avg_word_length = 6.2
            text = cess_esp.words()

        if language == 'english':   
            self.fdist = nltk.FreqDist(w for w in text)
        else:
            self.fdist = nltk.FreqDist(w.lower() for w in text)
        
        self.total = len(text)

        #models     
        self.model1 = MLPClassifier(random_state=2)
        self.model2 = svm.SVC(random_state=2)
        self.model4 = RandomForestClassifier(random_state=2) 
        self.model5 = LogisticRegression(random_state=2)

        #hard voting classifier 
        if language == 'spanish':
            estimators = [('mlp', self.model1), ('rf', self.model4), ('lr', self.model5)]
        else:
            estimators = [('svc', self.model2), ('rf', self.model4), ('mlp', self.model1)]

        self.vote = VotingClassifier(estimators, voting='hard')

    def extract_features(self, word):

      
        if len(word.split('-')) > 1:
            words = word.split('-')
        else:
            words = word.split(' ')

        len_tokens = len(words)
        freq_total = 0  
        chars_total = 0
        senses_total = 0 
        hypernym_total = 0  
        lemmas_total = 0 
        hypernym_total = 0 
        hyponym_total = 0
        syn_freq_total = 0 
        if len_tokens > 1:
            for word in words:
                lemmas = []
                hypernyms = []
                hyponyms = []
                freq_total += self.fdist[word]
                chars_total += len(word) / self.avg_word_length
                if self.language == 'english':
                    syn = wordnet.synsets(word)
                else:
                    syn = wordnet.synsets(word, lang='spa')
                senses_total += len(syn)
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

                lemmas_total+= len(lemmas)
                hypernym_total += len(hypernyms)
                hyponym_total += len(hyponyms)

            freq = freq_total/len_tokens
            len_chars = chars_total/ len_tokens
            syn_no = senses_total/ len_tokens
            lemma_no = lemmas_total/ len_tokens
            hypernym_no = hypernym_total/ len_tokens
            hyponym_no = hyponym_total/ len_tokens


            if freq == 0:
                len_freq = len_chars/1
            else:
                len_freq = len_chars/freq 
            

            lemma_freq = 0 
            for lemma in lemmas:
                lemma_freq += self.fdist[lemma]

                
            if lemma_freq > 0:
                if lemma_no > 0:
                    avg_lemma_freq = lemma_freq/lemma_no
                    syn_freq_total += freq/avg_lemma_freq
                else:
                    syn_freq_total += 0
            syn_freq = syn_freq_total / len_tokens


        else:

            freq = self.fdist[word]
            len_chars = len(word) / self.avg_word_length
            if self.language == 'english':
                syn = wordnet.synsets(word)
            else:
                syn = wordnet.synsets(word, lang='spa')
            syn_no = len(syn)

            lemmas = []
            hypernyms = []
            hyponyms = []
            vowel_count = 0 

            if freq == 0:
                len_freq = len(word)/1
            else:
                len_freq = len(word)/freq 

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

            vowels = {'a', 'e', 'i', 'o', 'u'}

            for letter in list(word):
                if letter in vowels:
                    vowel_count += 1 

            vowel_dif = 0 
            if vowel_count > 3:
                vowel_dif = 1
            else:
                vowel_dif = 0 

            lemma_freq = 0 
            for lemma in lemmas:
                lemma_freq += self.fdist[lemma]

            syn_freq = 0
            if lemma_freq > 0:
                if lemma_no > 0:
                    avg_lemma_freq = lemma_freq/lemma_no
                    syn_freq = freq/avg_lemma_freq
                else:
                    syn_freq = 0

        if self.language == 'english':
            return [len_freq, len_chars, len_tokens, freq, syn_no, lemma_no, hyponym_no, hypernym_no, syn_freq]
        else:
            return [len_freq, len_chars, len_tokens, freq, lemma_no, hyponym_no, hypernym_no, syn_freq] 


    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])
        
        self.model4.fit(X, y)


    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))
        
        return self.model4.predict(X)








