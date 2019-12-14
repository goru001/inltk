import asyncio
from math import ceil

from fastai.text import *
from inltk.config import LanguageCodes
from inltk.download_assets import setup_language, verify_language, check_all_languages_identifying_model
from inltk.tokenizer import LanguageTokenizer
from inltk.const import tokenizer_special_cases
from inltk.utils import cos_sim, reset_models

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


def get_similar_sentences(sen: str, no_of_variations: int, language_code: str):
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
    scores = []
    for word_vec in embedding_vectors:
        scores.append([cos_sim(word_vec, embdg) for embdg in embeddings])
    word_ids = [np.argpartition(-np.array(score), no_of_variations)[:no_of_variations] for score in scores]
    new_token_ids = []
    # generating more variations than required so that we can then filter out the best ones
    no_of_vars_per_token = ceil(no_of_variations/len(token_ids))*3
    for i in range(len(token_ids)):
        word_ids_list = word_ids[i].tolist()
        word_ids_list.remove(token_ids[i])
        for j in range(no_of_vars_per_token):
            new_token_ids.append(token_ids[:i] + word_ids_list[j:j+1] + token_ids[i+1:])
    new_sens = [tok.textify(tok_id) for tok_id in new_token_ids]
    while sen in new_sens:
        new_sens.remove(sen)
    sen_with_sim_score = [(new_sen, get_sentence_similarity(sen, new_sen, language_code)) for new_sen in new_sens]
    sen_with_sim_score.sort(key=lambda x: x[1], reverse=True)
    new_sens = [sen for sen, _ in sen_with_sim_score]
    return new_sens[:no_of_variations]
