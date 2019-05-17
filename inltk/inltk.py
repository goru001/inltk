import asyncio
from fastai.text import *
from .config import LanguageCodes
from .download_assets import setup_language, verify_language, check_all_languages_identifying_model
from inltk.tokenizer import LanguageTokenizer
from .const import tokenizer_special_cases

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
    path = Path(__file__).parent
    shutil.rmtree(path / 'models' / 'all')
    return


def get_embedding_vectors(input: str, language_code: str):
    check_input_language(language_code)
    tok = LanguageTokenizer(language_code)
    token_ids = tok.numericalize(input)
    # get learner
    defaults.device = torch.device('cpu')
    path = Path(__file__).parent
    learn = load_learner(path / 'models' / f'{language_code}')
    encoder = get_model(learn.model)[0]
    embeddings = encoder.state_dict()['encoder.weight']
    embeddings = np.array(embeddings)
    embedding_vectors = []
    for token in token_ids:
        embedding_vectors.append(embeddings[token])
    return embedding_vectors
