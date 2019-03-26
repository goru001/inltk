from pathlib import Path

import aiohttp as aiohttp
import os

from .config import LanguageCodes, LMConfigs

all_language_codes = LanguageCodes()

path = Path(__file__).parent


async def download_file(url, dest, fname):
    if (dest/f'{fname}').exists(): return
    os.makedirs(dest, exist_ok=True)
    print('Downloading Model... This might take time, depending on your internet connection! Please have patience!')
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest/f'{fname}', 'wb') as f:
                f.write(data)


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
