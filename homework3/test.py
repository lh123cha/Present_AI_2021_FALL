#使用word2vec将文本转化为向量

#使用word2vec将文本转化为向量
from __future__ import unicode_literals, print_function, division
import json
import gensim
import numpy as np
from lxml import etree
import string
import re
import unicodedata

import torch
import torch.nn as nn

from xml.dom.minidom import  parse
import xml.dom.minidom
from BLEU import bleu

MAX_LENGTH=20

def get_sentence(file):
    #读取文件
    rst={}
    domTree = parse(file)
    docs = domTree.getElementsByTagName("doc")

    for doc in docs:
        # print("docid",doc.getAttribute('docid'))
        docid=doc.getAttribute('docid')
        rst.setdefault(doc.getAttribute('docid'),{})
        segs = doc.getElementsByTagName('seg')
        for seg in segs:
            #print("id:%d,sentence:%s"%(int(seg.getAttribute('id')),seg.childNodes[0].data))
            rst[docid].setdefault(seg.getAttribute('id'), seg.childNodes[0].data)
    return  rst
def turn2txt():
    en_src=get_sentence(file=r"data\dev\dev\dev\newstest2015-ende-src.en.sgm")
    cn_ref=get_sentence(file=r"data\dev\dev\dev\newstest2015-ende-ref.de.sgm")
    with open(".//data//en_dev.txt","w",encoding='utf-8') as en_fp:
        with open(".//data//cn_dev.txt","w",encoding='utf-8') as cn_fp:
            for docid in en_src:
                for id in en_src[docid]:
                    if (docid in cn_ref) and (id in cn_ref[docid]):
                        print(en_src[docid][id])
                        print(cn_ref[docid][id])
                        en_fp.writelines(en_src[docid][id]+"\n")
                        cn_fp.writelines(cn_ref[docid][id]+"\n")
turn2txt()