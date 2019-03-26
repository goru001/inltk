import asyncio
from fastai.text import *
from .config import LanguageCodes
from .download_assets import setup_language, verify_language
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
