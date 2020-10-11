from fastai.text import *
import sentencepiece as spm
from pathlib import Path

from inltk.config import LanguageCodes
from inltk.utils import handle_all_caps, handle_upper_case_first_letter, lower_case_everything

path = Path(__file__).parent


class LanguageTokenizer(BaseTokenizer):
    def __init__(self, lang: str):
        self.lang = lang
        self.base = EnglishTokenizer(lang) if lang == LanguageCodes.english else IndicTokenizer(lang)

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
        model_path = path/f'models/{lang}/tokenizer.model'
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


class SanskritTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class BengaliTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class GujaratiTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class HindiTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class KannadaTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class MalyalamTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class MarathiTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class NepaliTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class OriyaTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class PanjabiTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class TamilTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class UrduTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class TeluguTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class HinglishTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        LanguageTokenizer.__init__(self, lang)


class TanglishTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        # because of some bug in fastai -- need to dive in further
        lang = LanguageCodes.tanglish
        LanguageTokenizer.__init__(self, lang)


class ManglishTokenizer(LanguageTokenizer):
    def __init__(self, lang: str):
        # because of some bug in fastai -- need to dive in further
        lang = LanguageCodes.manglish
        LanguageTokenizer.__init__(self, lang)
