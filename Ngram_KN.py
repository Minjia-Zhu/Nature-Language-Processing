
import math
from collections import Counter, defaultdict
import nltk
from nltk.corpus import brown
from nltk.util import ngrams
import string


def create_ngram(n, texts,test=False):
    '''
    input: list of sentence with word tokennize
    '''
    # Clean text and add pseudo start and end code
    lst_text = []
    for s in texts:
        s = [''.join(c for c in w if c not in string.punctuation) for w in s] # Remove punctuation
        s = [w.lower() for w in s if w] # Remove the empty strings
        s = ['<s>',] * (3 - 1) + s + ['</s>',]
        lst_text.extend(s)
    
    # For training text, include unknown words
    if not test:
        low_counter = [w for w in Counter(lst_text).keys() if Counter(lst_text)[w] < 2]
        for i, w in enumerate(lst_text):
            if w in low_counter :
                lst_text[i] = '<UNK>'

    # Build ngram
    ngrams = []
    for i in range(len(lst_text) - n + 1):
        ngram = tuple(lst_text)[i:i+n]
        ngrams.append(ngram)
    
    return ngrams

def classify(test,model_s,model_r,model_m):
    test = [s.split() for s in nltk.sent_tokenize(test)]
    ngram_test = create_ngram(3,test,True)
    s = model_s.score(ngram_test)
    m = model_m.score(ngram_test)
    r = model_r.score(ngram_test)
    d = {'mystery':m, 'romance': r, 'science fiction': s}
    winner = max(d, key=d.get)
    return winner

class NgramKN:

    def __init__(self,n,ngrams,d=0.75):
        self.n = n
        self.d = d
        self.fit = self.train(ngrams)

    def train(self, ngrams):
        all_counters = self._counts(Counter(ngrams))
        prob = self._probs(all_counters)
        return prob

    def _counts(self, highest_order_counter):
        # Normal count for the highest order gram
        all_counters = [highest_order_counter]
        
        # Continuation count for lower order grams
        for i in range(1, self.n):
            last_counter = all_counters[-1]
            new_counter = defaultdict(int)
            for ngram in last_counter.keys():
                suffix = ngram[1:]
                new_counter[suffix] += 1
            all_counters.append(new_counter)
        return all_counters

    def _probs(self, all_counters):
    
        backoffs = []
        
        # for non-unigram
        for counter in all_counters[:-1]:
            backoff = defaultdict(int)
            prefix_sums = defaultdict(int)
            
            for k in counter.keys():
                prefix = k[:-1]
                prefix_sums[prefix] += counter[k]
                counter[k] -= self.d
            
            for k in counter.keys():
                prefix = k[:-1]
                counter[k] = math.log(counter[k]/prefix_sums[prefix])
                
            for prefix in backoff.keys():
                backoff[prefix] = math.log(backoff[prefix]/prefix_sums[prefix])
 
            backoffs.append(backoff)
    
        # for unigram
        total_val = sum(v for v in all_counters[-1].values())
        all_counters[-1] = dict((k, math.log(v-self.d)/total_val) for k, v in all_counters[-1].items())
        backoffs.append(defaultdict(int))

        # Interpolation
        for last_order, order, backoff in zip(
                reversed(all_counters), reversed(all_counters[:-1]), reversed(backoffs[:-1])):
            for kgram in order.keys():
                prefix, suffix = kgram[:-1], kgram[1:]
                order[kgram] += last_order[suffix] + backoff[prefix]
            
        return all_counters


    def score(self, test_ngrams):
        logprob = 0
        for ngram in test_ngrams:
            for i, counter in enumerate(self.fit):
                if ngram[i:] in counter:
                    p = counter[ngram[i:]]
                else:
                    p = 0
                logprob += p           
        return logprob




