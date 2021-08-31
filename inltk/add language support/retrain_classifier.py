import fastai, torch
import os
import sentencepiece as spm

import pandas as pd
import re
from fastai.text import *
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import sentencepiece as spm
import re
import pdb

## Load Trained Tokenizer
sp = spm.SentencePieceProcessor()
sp.Load(str("all_language.model"))
itos = [sp.IdToPiece(int(i)) for i in range(150000)]

all_language_vocab = Vocab(itos)

class LanguageTokenizer(BaseTokenizer):
    def __init__(self, lang: str):
        self.lang = lang
        self.base = EnglishTokenizer(lang) if lang == 'en' else IndicTokenizer(lang)

    def tokenizer(self, t: str) -> List[str]:
        return self.base.tokenizer(t)

    def numericalize(self, t: str) -> List[int]:
        return self.base.numericalize(t)

    def textify(self, ids: List[int]) -> str:
        return self.base.textify(ids)

    def remove_foreign_tokens(self, t: str):
        return self.base.remove_foreign_tokens(t)


# Because we're using spacy tokenizer for english and sentence-piece for other languages
class EnglishTokenizer(BaseTokenizer):
    def __init__(self, lang: str):
        super().__init__(lang)
        self.lang = lang
        with open(path / f'models/{lang}/vocab.pkl', 'rb') as f:
            self.vocab = Vocab(pickle.load(f))
        self.tok = SpacyTokenizer(lang)

    def tokenizer(self, t: str) -> List[str]:
        tok = Tokenizer()
        tokens = tok.process_text(t, self.tok)
        tokens = [token for token in tokens if token not in defaults.text_spec_tok]
        return tokens

    def numericalize(self, t: str):
        token_ids = self.tokenizer(t)
        return self.vocab.numericalize(token_ids)

    def textify(self, ids: List[int]):
        return self.vocab.textify(ids)

    def remove_foreign_tokens(self, t: str):
        local_pieces = []
        for i in self.numericalize(t):
            local_pieces.append(self.textify([i]))
        return local_pieces


class IndicTokenizer(BaseTokenizer):
    def __init__(self, lang: str):
        self.lang = lang
        self.sp = spm.SentencePieceProcessor()
        model_path = 'all_language.model'  ## give path to trained tokenizer model
        self.sp.Load(str(model_path))

    def tokenizer(self, t: str) -> List[str]:
        return self.sp.EncodeAsPieces(t)

    def numericalize(self, t: str) -> List[int]:
        return self.sp.EncodeAsIds(t)

    def textify(self, ids: List[int]) -> str:
        return (''.join([self.sp.IdToPiece(id).replace('▁', ' ') for id in ids])).strip()

    def remove_foreign_tokens(self, t: str):
        local_pieces = []
        for i in self.sp.EncodeAsIds(t):
            local_pieces.append(self.sp.IdToPiece(i))
        return local_pieces


class AllLanguageTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)

tokenizer = Tokenizer(tok_func=AllLanguageTokenizer, lang='all')
tokenizer.special_cases

## Inititalise data loader
data_lm = TextLMDataBunch.from_csv(path = 'drive/My Drive', csv_name='all_languages.csv', text_cols=[1], label_cols=[2], tokenizer=tokenizer, vocab=all_language_vocab)
data_lm.batch_size = 16
data_lm.show_batch()

# Download fastai's language_model_learner
learn = language_model_learner(data_lm, AWD_LSTM,  drop_mult=0.1)
learn.lr_find()

learn.recorder.plot()

##First fit call
learn.fit_one_cycle(1, 1e-1, moms=(0.8,0.7))
#Second fit call (smaller learning rate)
learn.fit_one_cycle(2, 1e-3, moms=(0.8,0.7))

print(learn.predict('શાહરૂખ ખાન સાથે',n_words=10))

learn.save_encoder('all_language_enc')  ## save the trained encoder

##initialise classifier data loader
data_clas = TextClasDataBunch.from_csv(path=path, csv_name='all_languages.csv', tokenizer=tokenizer, vocab=all_language_vocab, text_cols=[0], label_cols=label_cols)

## downlaod fastai's text_classifier_learner
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('all_language_enc')
learn.freeze()
learn.lr_find()
learn.recorder.plot()

assert learn.loss_func.func == "CrossEntropyLoss()", "Change loss to cross entropy"

kappa = KappaScore()
learn.metrics = [kappa, accuracy]

# Training classifier
learn.fit_one_cycle(1, 1e-1, moms=(0.8,0.7))

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))

print(learn.predict('પ્રિયંકા ચોપડાને પતિ નિક પાસેથી મળી 2.7 કરોડ રૂપિયાની ગિફ્ટ, કિસ કરીને બોલી- લવ યૂ બેબી'))

learn.save('final')

defaults.device = torch.device('cpu')
learn.model.eval()
learn.export()