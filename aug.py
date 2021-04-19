import pandas as pd
import numpy as np
import torch

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action
from BackTranslation import BackTranslation
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SAMPLE_NUM = 1000
model_dir = 'model/'
des_dir = 'sentence/'
data_need_dir = des_dir + 'need_sentence.csv'
data_novel_dir = des_dir + 'novel_sentence.xlsx'
data_need_aug_dir = des_dir + 'need_aug_sentence.csv'
data_need_all_dir = des_dir + 'need_all_sentence.csv'

data_need = pd.read_csv(data_need_dir, index_col=0)

augs = [
    # Substitute word by word2vec similarity
    naw.WordEmbsAug(model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin',action="substitute"),
    # Substitute word by contextual word embeddings (BERT)
    naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute"),
    # Substitute word by WordNet's synonym
    naw.SynonymAug(aug_src='wordnet'),
    # Substitute word by PPDB's synonym
    naw.SynonymAug(aug_src='ppdb', model_path=model_dir + 'ppdb-2.0-s-all')
]

trans = BackTranslation(url=[
      'translate.google.com',
      #'translate.google.co.kr',
       'translate.google.cn',
    ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})
result = trans.translate('AUGMENTATION START:', src='en', tmp = 'zh-cn')
print(result.result_text)

data_need_aug = []
for text in tqdm(data_need['text']):
    for aug in augs:
        aug_text = aug.augment(text)
        data_need_aug.append(aug_text)
    res = trans.translate(text, src='en', tmp='zh-cn')
    data_need_aug.append(res.result_text)

df_need_aug = pd.DataFrame({'text': data_need_aug})
df_need_aug.to_csv(data_need_aug_dir)

data_need_aug = pd.read_csv(data_need_aug_dir, index_col=0)
data_need_aug_sample = data_need_aug.sample(n=SAMPLE_NUM, random_state=5)
df_need = pd.concat([data_need, data_need_aug_sample], ignore_index=True, sort=False)
df_need.to_csv(data_need_all_dir)
