import nltk
from nltk.corpus import wordnet as wn

def find_synsets(word):
    lst = set()
    synsets = wn.synsets(word)
    
    for synset in synsets:
        lemmas = synset.lemma_names()
        for lemma in lemmas:
            lst.add(lemma)
    return lst


if __name__ == '__main__':
    print(findSynsets('distasteful'))