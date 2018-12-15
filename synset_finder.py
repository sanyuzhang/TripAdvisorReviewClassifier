import nltk
from nltk.corpus import wordnet as wn


def find_synsets(word):
    return set(lemma for synset in wn.synsets(word) for lemma in synset.lemma_names())


def find_all_synsets(words):
    synsets = set()
    for w in words:
        synsets = synsets.union(find_synsets(w))
    return synsets


if __name__ == '__main__':
    s = find_all_synsets(['spacious', 'cozy'])
    print(s)