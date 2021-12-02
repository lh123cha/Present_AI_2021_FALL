#this file is used to caucultate the BLEU value of the translation
from __future__ import division

import math
import nltk
from nltk import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

def _modified_precision(candidate, references, n):
    counts = Counter(ngrams(candidate, n))

    if not counts:
        return 0

    max_counts = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n))
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    clipped_counts = dict((ngram, min(count, max_counts[ngram])) for ngram, count in counts.items())

    return sum(clipped_counts.values()) / sum(counts.values())
def _brevity_penalty(candidate, references):

    c = len(candidate)
    ref_lens = (len(reference) for reference in references)
    #这里有个知识点是Python中元组是可以比较的，如(0,1)>(1,0)返回False，这里利用元组比较实现了选取参考翻译中长度最接近候选翻译的句子，当最接近的参考翻译有多个时，选取最短的。例如候选翻译长度是10，两个参考翻译长度分别为9和11，则r=9.
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)
def bleu(candidate, references, weights):
    """Calculate BLEU score (Bilingual Evaluation Understudy)

    :param candidate: a candidate sentence
    :type candidate: list(str)
    :param references: reference sentences
    :type references: list(list(str))
    :param weights: weights for unigrams, bigrams, trigrams and so on
    :type weights: list(float)

    Papineni, Kishore, et al. "BLEU: A method for automatic evaluation of
    machine translation." Proceedings of the 40th annual meeting on association for
    computational linguistics. Association for Computational Linguistics, 2002.
    http://www.aclweb.org/anthology/P02-1040.pdf

    """
    p_ns = (
        _modified_precision(candidate, references, i)
        for i, _ in enumerate(weights, start=1)
    )

    try:
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns))
    except ValueError:
        # some p_ns is 0
        return 0

    bp = _brevity_penalty(candidate, references)
    return bp * math.exp(s)

# def ngrams(text,n):
#     words=[]
#     words=text.split(" ")
#     n_grams=[]
#     for i in range(1,n):
#         temp_gram=[]
#         for j in range(len(words)):
#             temp_gram.append(words[j:j+i])
#         n_grams.append(temp_gram)
#     return n_grams

weights=[1]
candidate1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
             'ensures', 'that', 'the', 'military', 'always',
               'obeys', 'the', 'commands', 'of', 'the', 'party']
candidate2 = ['It', 'is', 'to', 'insure', 'the', 'troops',
                   'forever', 'hearing', 'the', 'activity', 'guidebook',
                   'that', 'party', 'direct']
reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
                   'ensures', 'that', 'the', 'military', 'will', 'forever',
                   'heed', 'Party', 'commands']
reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
                   'guarantees', 'the', 'military', 'forces', 'always',
                   'being', 'under', 'the', 'command', 'of', 'the',
                   'Party']
reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
                   'army', 'always', 'to', 'heed', 'the', 'directions',
                   'of', 'the', 'party']