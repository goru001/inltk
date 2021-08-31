import fastai, torch
import os
import sentencepiece as spm

fastai.__version__

import pandas as pd
import re


############### LOAD TELUGU DATA (Technically can be any language that needs support)  ################################################

"""LOAD TELUGU DATA"""

df_wiki_telugu = pd.read_parquet("drive/My Drive/telugu_wiki_train.parquet")
df_wiki_telugu_test = pd.read_parquet("drive/My Drive/telugu_wiki_test.parquet")
df_wiki = pd.concat([df_wiki_telugu, df_wiki_telugu_test])

print(df_wiki.head())

df_telugu_news = pd.read_parquet("drive/My Drive/telugu_news_test (1).parquet")
df_telugu_news_train2 = pd.read_parquet("drive/My Drive/telugu_news_train (1).parquet")
df_news = pd.concat([df_telugu_news, df_telugu_news_train2])


text_wiki = df_wiki['text'].to_list()
text_news = df_news['text'].to_list()

def preprocess(article):
    article = re.sub(r'^https?:\/\/.*[\r\n]*', '', article)
    article = article.replace(u'\ufeff',' ')
    article = article.replace(u'\xa0', u' ')
    article = article.replace('  ', ' ');
    article = article.replace(' , ', ', ');
    return article

text = text_wiki + text_news
len(text)

sentences = []
for t in text:
    t = t.strip('\\n')
    ts = t.split('\n')
    ts = [s.strip() for s in ts if len(s) > 1]
    ts = [preprocess(s) for s in ts]
    sentences.extend(ts)
print(len(sentences))

sentences = [sentence for sentence in sentences if len(sentence.split()) > 2]
sentences = sentences[:200000]

len(sentences)

#### CREATE TELUGU TEXT FILE #########################
tel_path = os.path.join('./', 'telugu.txt')
with open(tel_path, 'w') as f:
    f.write('\n'.join(sentences))
#Train language specific Tokenizer
spm.SentencePieceTrainer.Train(f'--input={tel_path} --model_prefix=telugu_tok --vocab_size=25000')


##CREATING TEXT FILES FOR ALL LANGUAGES TO RETRAIN TOKENIZER
f = open("tamil.txt", 'w')
f = open("panjabi.txt", 'w')
f = open("sanskrit.txt", 'w')
f = open("nepali.txt", 'w')
f = open("malyalam.txt", 'w')
f = open("urdu.txt", 'w')
f = open("gujarati.txt", 'w')
f = open("marathi.txt", 'w')
f = open("hindi.txt", 'w')
f = open("odia.txt", 'w')
f = open("bengali.txt", 'w')
f = open("kannada.txt", 'w')
f = open("maithili.txt", 'w')

### Populating text files

df = pd.read_csv('all_languages.csv')

for index, row in df.iterrows():
  print(index)
  print(row['text'], row['label'])
  PATH = os.path.join(os.getcwd(),row['label'])
  PATH += '.txt'
  with open(PATH, 'a') as f:
    f.writelines(row['text'])


df['label'].value_counts()
labels = df['label'].unique()
labels =labels.tolist()
labels.append('telugu')

print(labels)

flist = []
for label in labels:
    flist.append(os.path.join(os.getcwd(), label) + ".txt")

flist = ','.join(flist)



#### TRAINING ALL LANGUAGE TOKENIZER #################
spm.SentencePieceTrainer.Train(f'--input={flist} --model_prefix=all_language --vocab_size=150000')

"""fastai saves the trained tokenizer in the same directory by default. So here the trained tokenizer would be
saved by the name all_language.model and vocab will get saved as all_language.vocab """