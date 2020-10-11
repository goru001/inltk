import sys
import warnings
import asyncio
import random
from math import ceil
from sklearn.metrics.pairwise import cosine_similarity

from fastai.text import *
from inltk.config import LanguageCodes
from inltk.download_assets import setup_language, verify_language, check_all_languages_identifying_model
from inltk.tokenizer import LanguageTokenizer
from inltk.const import tokenizer_special_cases
from inltk.utils import cos_sim, reset_models, is_english

if not sys.warnoptions:
    warnings.simplefilter("ignore")

lcodes = LanguageCodes()
all_language_codes = lcodes.get_all_language_codes()


async def download(language_code: str):
    if language_code not in all_language_codes:
        raise Exception(f'Language code should be one of {all_language_codes} and not {language_code}')
    learn = await setup_language(language_code)
    return learn


def setup(language_code: str):
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    tasks = [asyncio.ensure_future(download(language_code))]
    learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
    loop.close()


def check_input_language(language_code: str):
    if language_code not in all_language_codes:
        raise Exception(f'Language code should be one of {all_language_codes} and not {language_code}')
    if not verify_language(language_code):
        raise Exception(f'You need to do setup for the **first time** for language of your choice so that '
                        f'we can download models. So, '
                        f'Please run setup({language_code}) first!')


def predict_next_words(input: str, n_words: int, language_code: str, randomness=0.8):
    check_input_language(language_code)
    defaults.device = torch.device('cpu')
    path = Path(__file__).parent
    learn = load_learner(path / 'models' / f'{language_code}')
    output = learn.predict(input, n_words, randomness)
    # UTF-8 encoding takes care of both LTR and RTL languages
    if language_code != LanguageCodes.english:
        output = input + (''.join(output.replace(input, '').split(' '))).replace('▁', ' ')
    for special_str in tokenizer_special_cases:
        output = output.replace(special_str, '\n')
    return output


def tokenize(input: str, language_code: str):
    check_input_language(language_code)
    tok = LanguageTokenizer(language_code)
    output = tok.tokenizer(input)
    return output


def identify_language(input: str):
    if is_english(input):
        return 'en'
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    tasks = [asyncio.ensure_future(check_all_languages_identifying_model())]
    done = loop.run_until_complete(asyncio.gather(*tasks))[0]
    loop.close()
    defaults.device = torch.device('cpu')
    path = Path(__file__).parent
    learn = load_learner(path / 'models' / 'all')
    output = learn.predict(input)
    return str(output[0])


def remove_foreign_languages(input: str, host_language_code: str):
    check_input_language(host_language_code)
    tok = LanguageTokenizer(host_language_code)
    output = tok.remove_foreign_tokens(input)
    return output


def reset_language_identifying_models():
    reset_models('all')


def get_embedding_vectors(input: str, language_code: str):
    check_input_language(language_code)
    tok = LanguageTokenizer(language_code)
    token_ids = tok.numericalize(input)
    # get learner
    defaults.device = torch.device('cpu')
    path = Path(__file__).parent
    learn = load_learner(path / 'models' / f'{language_code}')
    encoder = get_model(learn.model)[0]
    encoder.reset()
    embeddings = encoder.state_dict()['encoder.weight']
    embeddings = np.array(embeddings)
    embedding_vectors = []
    for token in token_ids:
        embedding_vectors.append(embeddings[token])
    return embedding_vectors


def get_sentence_encoding(input: str, language_code: str):
    check_input_language(language_code)
    tok = LanguageTokenizer(language_code)
    token_ids = tok.numericalize(input)
    # get learner
    defaults.device = torch.device('cpu')
    path = Path(__file__).parent
    learn = load_learner(path / 'models' / f'{language_code}')
    encoder = learn.model[0]
    encoder.reset()
    kk0 = encoder(Tensor([token_ids]).to(torch.int64))
    return np.array(kk0[0][-1][0][-1])


def get_sentence_similarity(sen1: str, sen2: str, language_code: str, cmp: Callable = cos_sim):
    check_input_language(language_code)
    enc1 = get_sentence_encoding(sen1, language_code)
    enc2 = get_sentence_encoding(sen2, language_code)
    return cmp(enc1, enc2)


def get_similar_sentences(sen: str, no_of_variations: int, language_code: str, degree_of_aug: float = 0.1):
    check_input_language(language_code)
    # get embedding vectors for sen
    tok = LanguageTokenizer(language_code)
    token_ids = tok.numericalize(sen)
    embedding_vectors = get_embedding_vectors(sen, language_code)
    # get learner
    defaults.device = torch.device('cpu')
    path = Path(__file__).parent
    learn = load_learner(path / 'models' / f'{language_code}')
    encoder = get_model(learn.model)[0]
    encoder.reset()
    embeddings = encoder.state_dict()['encoder.weight']
    embeddings = np.array(embeddings)
    # cos similarity of vectors
    scores = cosine_similarity(embedding_vectors,embeddings)
    word_ids = [np.argpartition(-np.array(score), no_of_variations+1)[:no_of_variations+1] for score in scores]
    word_ids = [ids.tolist() for ids in word_ids]
    for i, ids in enumerate(word_ids):
        word_ids[i] = [wid for wid in word_ids[i] if wid != token_ids[i]]
    # generating more variations than required so that we can then filter out the best ones
    buffer_multiplicity = 2
    new_sen_tokens = []
    for i in range(no_of_variations):
        for k in range(buffer_multiplicity):
            new_token_ids = []
            ids = sorted(random.sample(range(len(token_ids)), max(1, int(degree_of_aug * len(token_ids)))))
            for j in range(len(token_ids)):
                if j in ids:
                    new_token_ids.append(word_ids[j][(i + k) % len(word_ids[j])])
                else:
                    new_token_ids.append(token_ids[j])
            new_token_ids = list(map(lambda x: int(x), new_token_ids))
            new_sen_tokens.append(new_token_ids)
    new_sens = [tok.textify(sen_tokens) for sen_tokens in new_sen_tokens]
    while sen in new_sens:
        new_sens.remove(sen)
    sen_with_sim_score = [(new_sen, get_sentence_similarity(sen, new_sen, language_code)) for new_sen in new_sens]
    sen_with_sim_score.sort(key=lambda x: x[1], reverse=True)
    new_sens = [sen for sen, _ in sen_with_sim_score]
    return new_sens[:no_of_variations]
