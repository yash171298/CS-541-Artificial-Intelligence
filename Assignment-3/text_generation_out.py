#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import random
import time
from typing import List


# In[2]:


def tokenize(text: str) -> List[str]:
    for punct in string.punctuation:
        text = text.replace(punct, ' '+punct+' ')
    t = text.split()
    return t

def get_ngrams(n: int, tokens: list) -> list:
    tokens = (n-1)*['<START>']+tokens
    l = [(tuple([tokens[i-p-1] for p in reversed(range(n-1))]), tokens[i]) for i in range(n-1, len(tokens))]
    return l


# In[6]:


class NgramModel(object):

    def __init__(self, n):
        self.n = n
        self.context = {}
        self.ngram_counter = {}

    def update(self, sentence: str) -> None:
        n = self.n
        ngrams = get_ngrams(n, tokenize(sentence))
        for ngram in ngrams:
            if ngram in self.ngram_counter:
                self.ngram_counter[ngram] += 1.0
            else:
                self.ngram_counter[ngram] = 1.0

            prev_words, target_word = ngram
            if prev_words in self.context:
                self.context[prev_words].append(target_word)
            else:
                self.context[prev_words] = [target_word]
                
    def freq(self, context, token):
        try:
            count_of_token = self.ngram_counter[(context, token)]
            count_of_context = float(len(self.context[context]))
            result = count_of_token / count_of_context

        except KeyError:
            result = 0.0
        return result

    def random_token(self, context):
        r = random.random()
        map_to_probs = {}
        token_of_interest = self.context[context]
        for token in token_of_interest:
            map_to_probs[token] = self.freq(context, token)

        summ = 0
        for token in sorted(map_to_probs):
            summ += map_to_probs[token]
            if summ > r:
                return token

    def generate_text(self, token_count: int):
        n = self.n
        context_queue = (n - 1) * ['<START>']
        result = []
        for _ in range(token_count):
            obj = self.random_token(tuple(context_queue))
            result.append(obj)
            if n > 1:
                context_queue.pop(0)
                if obj == '.':
                    context_queue = (n - 1) * ['<START>']
                else:
                    context_queue.append(obj)
        return ' '.join(result)


# In[22]:


def create_ngram_model(n, path):
    m = NgramModel(n)
    with open(path, 'r', encoding="utf8") as f:
        text = f.read()
        text = text.split('.')
        for sentence in text:
            # add back the fullstop
            sentence += '.'
            m.update(sentence)
    return m


# In[32]:


if __name__ == "__main__":
    start = time.time()
    m = create_ngram_model(2, 'reddit.txt')
    print (f'Time taken: {time.time() - start}')
    start = time.time()
    random.seed(7)
    x = m.generate_text(500)
    print(f'Generated text:'+ x)
    print(m.generate_text(500))
    print(m.generate_text(500),file=open("reddit.out", "a", encoding="utf8"))


# In[33]:


if __name__ == "__main__":
    start = time.time()
    m = create_ngram_model(3, 'ted.txt')

    print (f'Time taken: {time.time() - start}')
    start = time.time()
    random.seed(7)
    x = m.generate_text(500)
    print(f'Generated text:'+ x)
    print(m.generate_text(500))
    print(m.generate_text(500),file=open("ted.out", "a", encoding="utf8"))


# In[ ]:




