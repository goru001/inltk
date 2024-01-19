import string
from unidecode import unidecode
import codecs
import binascii

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from typing import Iterable, List

import os
import random
import re
import numpy as np
from collections import OrderedDict

from requests.exceptions import ConnectionError
import nltk
import stanza
try:
    nltk.download('punkt')
    stanza.download('en')
    #stanza.download('ta')
    en_nlp = stanza.Pipeline('en', processors='tokenize')
    #ta_nlp = stanza.Pipeline('ta', processors='tokenize')
except ConnectionError:
    en_nlp = stanza.Pipeline('en', processors='tokenize', download_method=None)
    #ta_nlp = stanza.Pipeline('ta', processors='tokenize', download_method=None)

from utils.dataset_visualization import visualize_dataset_for_bucketing_stats

#from gensim.models.fasttext import load_facebook_model
from gensim.models import Word2Vec

from datasets.utils import return_unicode_hex_within_range, return_tamil_unicode_isalnum, check_unicode_block

class EnTamV2Dataset(Dataset):

    SRC_LANGUAGE = 'en'
    TGT_LANGUAGE = 'ta'

    # Using START and END tokens in source and target vocabularies to enforce better relationships between x and y
    reserved_tokens = ["UNK", "PAD", "START", "END", "NUM", "ENG"]
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, NUM_IDX, ENG_IDX = 0, 1, 2, 3, 4, 5
    num_token_sentences = 500

    word_vector_size = 100
  
    # Pretrained word2vec models take too long to load (many hours) - training my own
    #en_wv = load_facebook_model('dataset/monolingual/cc.en.300.bin_fasttext.bin')
    #ta_wv = load_facebook_model('dataset/monolingual/cc.ta.300.bin_fasttext.gz')

    def __init__(self, split, symbols=False, verbose=False):
        
        # NOTE: symbols = False and True both use the same file naming conventions. Move the cached files accordingly

        self.verbose = verbose

        tokenized_dirname = "tokenized"
        if not os.path.exists(self.get_dataset_filename(split, "en", tokenized_dirname)) \
                or not os.path.exists(self.get_dataset_filename(split, "ta", tokenized_dirname)):
            
            self.bilingual_pairs, eng_words = self.get_sentence_pairs(split, symbols=symbols)
            
            if split == "train":
                eng_words = list(eng_words)
                self.create_token_sentences_for_word2vec(eng_words)

            self.eng_vocabulary, self.eng_word_counts, tokenized_eng_sentences = self.create_vocabulary([
                                                                                    x[0] for x in self.bilingual_pairs], language="en")
            self.tam_vocabulary, self.tam_word_counts, tokenized_tam_sentences = self.create_vocabulary([
                                                                                    x[1] for x in self.bilingual_pairs], language="ta")

            if self.verbose:
                print ("Most Frequent 1000 English tokens:", sorted(self.eng_word_counts, key=lambda y: self.eng_word_counts[y], reverse=True)[:1000])
                print ("Most Frequent 1000 Tamil tokens:", sorted(self.tam_word_counts, key=lambda y: self.tam_word_counts[y], reverse=True)[:1000])

            # save tokenized sentences for faster loading

            with open(self.get_dataset_filename(split, "en", tokenized_dirname), 'w') as f:
                for line in tokenized_eng_sentences:
                    f.write("%s\n" % line)
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname), 'w') as f:
                for line in tokenized_tam_sentences:
                    f.write("%s\n" % line)
            with open(self.get_dataset_filename(split, "en", tokenized_dirname, substr="vocab"), 'w') as f:
                for word in self.eng_vocabulary:
                    f.write("%s\n" % word)
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname, substr="vocab"), 'w') as f:
                for word in self.tam_vocabulary:
                    f.write("%s\n" % word)
            self.bilingual_pairs = list(zip(tokenized_eng_sentences, tokenized_tam_sentences))
        
        else:
            
            with open(self.get_dataset_filename(split, "en", tokenized_dirname), 'r') as f:
                tokenized_eng_sentences = [x.strip() for x in f.readlines()]
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname), 'r') as f:
                tokenized_tam_sentences = [x.strip() for x in f.readlines()]
            with open(self.get_dataset_filename(split, "en", tokenized_dirname, substr="vocab"), 'r') as f:
                self.eng_vocabulary = [x.strip() for x in f.readlines()]
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname, substr="vocab"), 'r') as f:
                self.tam_vocabulary = [x.strip() for x in f.readlines()]
            self.bilingual_pairs = list(zip(tokenized_eng_sentences, tokenized_tam_sentences))
        
        assert "DEBUG" not in self.eng_vocabulary, "Debug token found in final train dataset"

        print ("English vocabulary size for %s set: %d" % (split, len(self.eng_vocabulary)))
        print ("Tamil vocabulary size for %s set: %d" % (split, len(self.tam_vocabulary)))
        
        print ("Using %s set with %d sentence pairs" % (split, len(self.bilingual_pairs)))

        if not os.path.exists('utils/Correlation.png') and split == "train":
            visualize_dataset_for_bucketing_stats(self.bilingual_pairs)
        
        if not os.path.exists("dataset/word2vec/word2vec_entam.en.model") or not \
                os.path.exists("dataset/word2vec/word2vec_entam.ta.model"):
            if split == "train":
                self.train_word2vec_model_on_monolingual_and_mt_corpus(symbols, \
                        tokenized_eng_sentences, tokenized_tam_sentences)
        
        else:
            
            if self.verbose:
                print ("Loading trained word2vec models")
            self.en_wv = Word2Vec.load("dataset/word2vec/word2vec_entam.en.model")
            self.ta_wv = Word2Vec.load("dataset/word2vec/word2vec_entam.ta.model")
        
        # Sanity check for word vectors OOV
        # DEBUG: Commenting for training dataset
        """
        for sentence in tokenized_eng_sentences:
            for token in sentence.split(' '):
                self.get_word2vec_embedding_for_token(token, split, "en")
        for sentence in tokenized_tam_sentences:
            for token in sentence.split(' '):
                self.get_word2vec_embedding_for_token(token, split, "ta")
        """
    
    def __len__(self):
        return len(self.bilingual_pairs)

    def __getitem__(self, idx):

        #TODO: Manual bucketing implementation

    def get_word2vec_embedding_for_token(self, token, split, lang="en"):
        
        try:
            if lang == "en":
                return self.en_wv.wv[token]
            else:
                return self.ta_wv.wv[token]
        
        except KeyError:

            if self.verbose:
                print ("Token not in %s word2vec vocabulary: %s" % (split, token))
            # word vector not in vocabulary - possible for tokens in val and test sets
            return np.random.rand(self.word_vector_size)

    def train_word2vec_model_on_monolingual_and_mt_corpus(self, symbols, en_train_set, ta_train_set):

        with open(self.get_dataset_filename("train", "en", subdir="word2vec", substr="word2vec"), 'r') as f:
            eng_word2vec = [x.strip() for x in f.readlines()]

        with open(self.get_dataset_filename("train", "ta", subdir="word2vec", substr="word2vec"), 'r') as f:
            tam_word2vec = [x.strip() for x in f.readlines()]
        
        if self.verbose:
            print ("Preprocessing word2vec datasets for English and Tamil")
        word2vec_sentences, word2vec_eng_words = self.get_sentence_pairs("train", symbols=symbols, dataset=[eng_word2vec, tam_word2vec])
        en_word2vec, ta_word2vec = word2vec_sentences
        
        if self.verbose:
            print ("Tokenizing word2vec English and Tamil monolingual corpora")
        _,_, en_word2vec = self.create_vocabulary(en_word2vec, language="en")
        _,_, ta_word2vec = self.create_vocabulary(ta_word2vec, language="ta")
        
        en_word2vec.extend(en_train_set)
        ta_word2vec.extend(ta_train_set)

        en_word2vec = [x.split(' ') for x in en_word2vec]
        ta_word2vec = [x.split(' ') for x in ta_word2vec]
        
        if self.verbose:
            print ("Training word2vec vocabulary for English")
        self.en_wv = Word2Vec(sentences=en_word2vec, vector_size=word_vector_size, window=5, min_count=1, workers=4)
        self.en_wv.build_vocab(en_word2vec)
        self.en_wv.train(en_word2vec, total_examples=len(en_word2vec), epochs=20)
        self.en_wv.save("dataset/word2vec/word2vec_entam.en.model")

        if self.verbose:
            print ("Training word2vec vocabulary for Tamil")
        self.ta_wv = Word2Vec(sentences=ta_word2vec, vector_size=word_vector_size, window=5, min_count=1, workers=4)
        self.ta_wv.build_vocab(ta_word2vec)
        self.ta_wv.train(ta_word2vec, total_examples=len(ta_word2vec), epochs=20)
        self.ta_wv.save("dataset/word2vec/word2vec_entam.ta.model")

    def get_dataset_filename(self, split, lang, subdir=None, substr=""): 
        assert split in ['train', 'dev', 'test', ''] and lang in ['en', 'ta', ''] # Using '' to get dirname because dataset was defined first here!
        assert substr in ["", "vocab", "word2vec"]
        if not subdir is None:
            if substr not in ["vocab", "word2vec"]:
                directory = os.path.join("dataset", subdir, "%s.bcn" % "corpus")
            else:
                directory = os.path.join("dataset", subdir, "%s.bcn" % substr)
        else:
            if substr not in ["vocab", "word2vec"]:
                directory = os.path.join("dataset", "%s.bcn" % "corpus")
            else:
                raise AssertionError

        full_path = "%s.%s.%s" % (directory, split, lang)
        
        save_dir = os.path.dirname(full_path)
        
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        
        return full_path

    def get_sentence_pairs(self, split, symbols=False, dataset=None):
        # use symbols flag to keep/remove punctuation

        text_pairs = []
        translator = str.maketrans('', '', string.punctuation)
        
        unnecessary_symbols = ["‘", "¦", "¡", "¬", '“', '”', "’", '\u200c'] # negation symbol might not be in EnTamV2
        # Exclamation mark between words in train set
        
        if symbols:
            symbol_replacements = {unnecessary_symbols[0]: "'", unnecessary_symbols[4]: '"', unnecessary_symbols[5]: "\"", unnecessary_symbols[6]: "'"}
        else:
            symbol_replacements = {}
        
        if not dataset is None:
            eng_sentences = dataset[0]
        else:
            with open(self.get_dataset_filename(split, self.SRC_LANGUAGE), 'r') as l1:
                eng_sentences = [re.sub(
                                    '\d+', ' %s ' % self.reserved_tokens[self.NUM_IDX], x.lower() # replace all numbers with [NUM] token
                                    ).strip().replace("  ", " ")
                                            for x in l1.readlines()]
            
        for idx, sentence in enumerate(eng_sentences):
            
            for sym in symbol_replacements:
                eng_sentences[idx] = eng_sentences[idx].replace(sym, " " + symbol_replacements[sym] + " ")
            
            if not symbols:
                eng_sentences[idx] = re.sub(r'[^\w\s*]|_',r' ', sentence)
            else:
                # couldn't use re here, not sure why
                for ch in string.punctuation:
                    eng_sentences[idx] = eng_sentences[idx].replace(ch, " "+ch+" ")

            for sym_idx, sym in enumerate(unnecessary_symbols):
                #return_in_bilingual_corpus = eng_sentences[idx]
                if not symbols or (symbols and not sym in symbol_replacements.keys()):
                    eng_sentences[idx] = eng_sentences[idx].replace(sym, "")

            eng_sentences[idx] = re.sub("\s+", " ", eng_sentences[idx]) # correct for number of spaces
            
            # manual corrections
            eng_sentences[idx] = eng_sentences[idx].replace("naa-ve", "naive")
            eng_sentences[idx] = re.sub(r"j ' (\w)", r"i \1", eng_sentences[idx])
            eng_sentences[idx] = eng_sentences[idx].replace(". . .", "...")
        
        if not dataset is None:
            tam_sentences_file = dataset[1]
        else:
            with open(self.get_dataset_filename(split, self.TGT_LANGUAGE), 'r') as l2:
                # 2-character and 3-character alphabets are not \w (words) in re, switching to string.punctuation

                tam_sentences_file = list(l2.readlines())
            
        eng_words, tam_sentences = set(), []
        for idx, sentence in enumerate(tam_sentences_file):
        
            # some english words show up in tamil dataset (lower case)
            line = re.sub('\d+', ' %s ' % self.reserved_tokens[self.NUM_IDX], sentence.lower()) # use NUM reserved token

            if not symbols:
                line = line.translate(translator) # remove punctuations
            else:
                # couldn't use re here, not sure why
                for ch in string.punctuation:
                    line = line.replace(ch, " "+ch+" ")
                
                for sym in symbol_replacements:
                    line = line.replace(sym, " "+symbol_replacements[sym]+" ")

            for sym in unnecessary_symbols:
                if not symbols or (symbols and not sym in symbol_replacements.keys()):
                    line = line.replace(sym, "") 
            
            line = re.sub("\s+", " ", line) # correct for number of spaces
            
            if dataset is None:
                p = re.compile("([a-z]+)\s|([a-z]+)")
                search_results = p.search(line)
                if not search_results is None:
                    eng_tokens = [x for x in search_results.groups() if not x is None]
                    eng_words.update(eng_tokens)

                    with open(self.get_dataset_filename("train", "en", subdir="tamil_eng_vocab_untokenized"), 'a') as f:
                        f.write("%s\n" % (eng_sentences[idx]))
                    with open(self.get_dataset_filename("train", "ta", subdir="tamil_eng_vocab_untokenized"), 'a') as f:
                        f.write("%s\n" % (sentence))

            line = re.sub("[a-z]+\s*", "%s " % self.reserved_tokens[self.ENG_IDX], line) # use ENG reserved token
            
            line = line.replace(". . .", "...")
            tam_sentences.append(line.strip())
        
        if dataset is None:
            for eng, tam in zip(eng_sentences, tam_sentences):
                text_pairs.append((eng, tam))
        
            random.shuffle(text_pairs)
        
            return text_pairs, eng_words
        
        else: #word2vec
            
            return [eng_sentences, tam_sentences], eng_words

    def create_token_sentences_for_word2vec(self, eng_words):
        
        # DEBUG
        # tamil sentence has no english words for transfer to english vocabulary
        if len(eng_words) == 0:
            eng_words = ["DEBUG"]

        # instantiate for train set only
        self.eng_words = eng_words

        if len(eng_words) < self.num_token_sentences:
            eng_words = list(np.tile(eng_words, self.num_token_sentences//len(eng_words) + 1)[:self.num_token_sentences])

        self.reserved_token_sentences = []
        for idx in range(len(eng_words)):
            string="%s " % self.reserved_tokens[self.BOS_IDX]
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.NUM_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.NUM_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += "%s " % self.reserved_tokens[self.EOS_IDX]
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string = string.strip()
            
            src_string = string.replace(self.reserved_tokens[self.UNK_IDX], eng_words[idx])
            trg_string = string.replace(eng_words[idx], self.reserved_tokens[self.ENG_IDX])
            self.reserved_token_sentences.append((src_string, trg_string))

    def create_vocabulary(self, sentences, language='en'):
        
        assert language in ['en', 'ta']

        for idx in range(len(sentences)):
            if language == 'en':
                sentences[idx] = unidecode(sentences[idx]) # remove accents from english sentences
                # Refer FAQs here: https://pypi.org/project/Unidecode/
                sentences[idx] = sentences[idx].replace("a<<", "e") # umlaut letter
                sentences[idx] = sentences[idx].replace("a 1/4", "u") # u with diaeresis
                sentences[idx] = sentences[idx].replace("a3", "o") # ó: a3 --> o
                
                sentences[idx] = sentences[idx].replace("a(r)", "i") # î: a(r) --> i
                sentences[idx] = sentences[idx].replace("a-", "i") # ï: a- --> i [dataset seems to also use ocr: ïடக்கக்கூடியதுதான்  --> i(da)kka for padi[kka]]
                sentences[idx] = sentences[idx].replace("a$?", "a") # ä: a$? --> a
                sentences[idx] = sentences[idx].replace("a'", "o") # ô: a' --> o
                sentences[idx] = sentences[idx].replace("d1", "e") # econostrum - single token
                sentences[idx] = sentences[idx].replace("a+-", "n") # ñ: a+- --> n
                sentences[idx] = sentences[idx].replace("a1", "u") # ù: a1 --> u
            
                # manual change
                num_and_a_half = lambda x: "%s%s" % (self.reserved_tokens[self.NUM_IDX], x) # NUM a half --> NUM and a half
                sentences[idx] = sentences[idx].replace(num_and_a_half(" a 1/2"), num_and_a_half(" and a half"))
            
            sentences[idx] = self.reserved_tokens[self.BOS_IDX] + ' ' + sentences[idx] + ' ' + self.reserved_tokens[self.EOS_IDX]
            sentences[idx] = re.sub('\s+', ' ', sentences[idx])

        if hasattr(self, "reserved_token_sentences"):
            if language == 'en':
                sentences.extend([x[0] for x in self.reserved_token_sentences])
            elif language == 'ta':
                sentences.extend([x[1] for x in self.reserved_token_sentences])
        
        # English
        # 149309 before tokenization ; 75210 after
        # 70765 tokens without symbols
        # 67016 tokens without numbers
        
        # nltk vs stanza: 66952 vs 66942 tokens

        # Tamil
        # 271651 tokens with English words
        # 264429 tokens without English words (ENG tag)
        
        vocab = set()
        word_counts = {}
        
        virama_introduction_chars = {"ங": "ங்"}

        if hasattr(self, "eng_tokens"):
            vocab.update(self.eng_tokens)

        for idx, sentence in enumerate(sentences):
            if idx == len(sentences) - self.num_token_sentences and hasattr(self, 'reserved_token_sentences'):
                vocab.update(self.reserved_tokens)
                break
            else:
                if language == 'en':
                    #tokens = nltk.word_tokenize(sentence)
                    doc = en_nlp(sentence)
                    if len(doc.sentences) > 1:
                        tokens = [x.text for x in doc.sentences[0].tokens]
                        for sent in doc.sentences[1:]:
                            tokens.extend([x.text for x in sent.tokens])
                    else:
                        tokens = [x.text for x in doc.sentences[0].tokens]
                    
                elif language == 'ta':
                    # stanza gives tokens of single alphabets that don't make semantic sense and increases vocab size
                    # Because of data preprocessing and special character removal, stanza doesn't do much for tokenizing tamil

                    tokens = sentence.split(' ')
                    
                    # sentence 32579: 
                    for token_index, token in enumerate(tokens):

                        if not token in string.punctuation:
                            
                            token_languages = self.get_entam_sequence(token)
                            token_replacement = self.tokenize_entam_combinations(token_languages, token)
                            
                            if token_replacement[0] == token:
                                continue
                            else:
                                new_sentence_tokens = tokens[:token_index] + token_replacement + tokens[token_index+1:]
                                sentences[idx] = " ".join(new_sentence_tokens)

                for token_idx, token in enumerate(tokens):
                    
                    # use stress character (virama from wikipedia) to end tokens that need them
                    if language == "ta" and token[-1] in virama_introduction_chars.keys():
                        token = token[:-1] + virama_introduction_chars[token[-1]]
                    
                    # many OOV words have accented english + ENG token, removing the accents using the reserved token
                    #if self.reserved_tokens[self.ENG_IDX] in token and len(token) > len(self.reserved_tokens[self.ENG_IDX]):
                    #    token = self.reserved_tokens[self.ENG_IDX]

                    if token in vocab:
                        word_counts[token] += 1
                    else:
                        word_counts[token] = 1
                
                vocab.update(tokens)
                sentences[idx] = " ".join(tokens)
                
        if hasattr(self, "eng_vocab"):
            if language == "en":
                tokens_in_eng_vocabulary = 4 # only UNK and ENG don't belong to en vocabulary
                assert len(word_counts) == len(vocab) - (len(self.reserved_tokens) - tokens_in_eng_vocabulary), \
                        "sentence %d: Vocab size: %d, Word Count dictionary size: %d" % (idx, len(vocab), len(word_counts)) # BOS, EOS, NUM, PAD already part of sentences
            else:
                assert len(word_counts) == len(vocab), \
                        "sentence %d: Vocab size: %d, Word Count dictionary size: %d" % (idx, len(vocab), len(word_counts)) # BOS, EOS, NUM, PAD, ENG already part of sentences

        return vocab, word_counts, sentences

    def tokenize_entam_combinations(self, token_languages, token):

        tokens_split, tamil_part = [], ""
        keys = list(token_languages.keys())
        for idx, key in enumerate(reversed(keys[:-1])):
            lang = "en" if "en" in key else "ta"
            start_of_lang_block = token_languages[key]
            end_of_lang_block = token_languages[keys[len(keys)-1 - idx]]
            
            if lang == "en":
                if end_of_lang_block - start_of_lang_block >= 3:
                    if tamil_part == "":
                        tokens_split.append(self.reserved_tokens[self.ENG_IDX])
                    else:
                        tokens_split.extend([tamil_part, self.reserved_tokens[self.ENG_IDX]])
                        tamil_part = ""
            else:
                tamil_part = token[start_of_lang_block:end_of_lang_block] + tamil_part
        
        if tamil_part != "":
            tokens_split.append(tamil_part)
        else:
            # no tamil characters means <=2 character english token
            tokens_split.append(self.reserved_tokens[self.ENG_IDX])

        tokens_split = list(reversed(tokens_split))

        return tokens_split
    
    def get_entam_sequence(self, token):
        
        if not hasattr(self, "tamil_characters_hex"):
            self.tamil_characters_hex = return_tamil_unicode_isalnum()
        
        sequence = OrderedDict()
        num_eng, num_tam = 0, 0
        get_count = lambda lang: str(num_eng) if lang=='en' else str(num_tam)

        if check_unicode_block(token[0], self.tamil_characters_hex):
            lang = 'ta'
            num_tam += 1
        else:
            lang = 'en'
            num_eng += 1

        sequence[lang+"0"] = 0

        for idx, character in enumerate(list(token)[1:]):

            if check_unicode_block(character, self.tamil_characters_hex):
                if lang == 'en':
                    lang = 'ta'
                    sequence[lang+get_count(lang)] = idx + 1
                    num_tam += 1
            else:
                if lang == 'ta':
                    lang = 'en'
                    sequence[lang+get_count(lang)] = idx + 1
                    num_eng += 1

        sequence[lang+get_count(lang)] = len(token)
        return sequence    

    def get_tamil_special_characters(self, sentence, idx):

        if not hasattr(self, "tamil_characters_hex"):
            self.tamil_characters_hex = return_tamil_unicode_isalnum()
        
        if sentence in self.reserved_tokens:
            return [], "", False

        spl_chars, tamil_token, prefix = [], "", False
        for unicode_2_or_3 in sentence:
            # token level special character search doesn't need to check for space
            #if unicode_2_or_3 == ' ':
            #    continue

            unicode_hex = "".join("{:02x}".format(ord(x)) for x in unicode_2_or_3)
            if not unicode_hex in self.tamil_characters_hex:
                if len(spl_chars) == 0:
                    prefix = True
                if not unicode_2_or_3 in string.punctuation:
                    spl_chars.append(unicode_2_or_3)
            else:
                tamil_token += unicode_2_or_3
        
        assert len(spl_chars) + len(tamil_token) == len(sentence), \
                "sentence %d: Complicated English-Tamil combo word: %s (%d), spl chars: %s (%d), tamil: %s (%d)" % (
                        idx, sentence, len(sentence), spl_chars, len(spl_chars), tamil_token, len(tamil_token))

        return spl_chars, tamil_token, prefix

if __name__ == "__main__":
    
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", help="Verbose flag for dataset stats", action="store_true")
    args = ap.parse_args()

    #train_dataset = EnTamV2Dataset("train", verbose=args.verbose)
    #val_dataset = EnTamV2Dataset("dev", verbose=args.verbose)
    #test_dataset = EnTamV2Dataset("test", verbose=args.verbose)

    train_dataset = EnTamV2Dataset("train", symbols=True, verbose=args.verbose)
    val_dataset = EnTamV2Dataset("dev", symbols=True, verbose=args.verbose)
    test_dataset = EnTamV2Dataset("test", symbols=True, verbose=args.verbose)

