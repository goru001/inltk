from pathlib import Path

import aiohttp as aiohttp
import os

from .config import LanguageCodes, LMConfigs, AllLanguageConfig

all_language_codes = LanguageCodes()

path = Path(__file__).parent


async def download_file(url, dest, fname):
    if (dest/f'{fname}').exists(): return False
    os.makedirs(dest, exist_ok=True)
    print('Downloading Model. This might take time, depending on your internet connection. Please be patient.\n'
          'We\'ll only do this for the first time.')
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest/f'{fname}', 'wb') as f:
                f.write(data)
    return True


async def setup_language(language_code: str):
    lmconfig = LMConfigs(language_code)
    config = lmconfig.get_config()
    await download_file(config['lm_model_url'], path/'models'/f'{language_code}', config["lm_model_file_name"])
    await download_file(config['tokenizer_model_url'], path/'models'/f'{language_code}',
                        config["tokenizer_model_file_name"])
    print('Done!')
    return True


def verify_language(language_code: str):
    lmconfig = LMConfigs(language_code)
    config = lmconfig.get_config()
    if (path/'models'/f'{language_code}'/f'{config["lm_model_file_name"]}').exists() and \
            (path/'models'/f'{language_code}'/f'{config["tokenizer_model_file_name"]}').exists():
        return True
    else:
        return False


async def check_all_languages_identifying_model():
    config = AllLanguageConfig.get_config()
    if (path/'models'/'all'/f'{config["all_languages_identifying_model_name"]}').exists() and \
            (path/'models'/'all'/f'{config["all_languages_identifying_tokenizer_name"]}').exists():
        return True
    done = await download_file(config["all_languages_identifying_model_url"], path/'models'/'all',
                        config["all_languages_identifying_model_name"])
    done = await download_file(config["all_languages_identifying_tokenizer_url"], path/'models'/'all',
                        config["all_languages_identifying_tokenizer_name"])
    return done

