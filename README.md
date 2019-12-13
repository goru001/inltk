## Natural Language Toolkit for Indic Languages (iNLTK)

[![Gitter](https://badges.gitter.im/inltk/community.svg)](https://gitter.im/inltk/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

iNLTK aims to provide out of the box support for various NLP tasks 
that an application developer might need for Indic languages.

![Alt Text](inltk/static/inltk.gif)

### Installation on Linux

```bash
pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install inltk
``` 

Note: Just make sure to pick the correct torch wheel url, according to the needed 
platform and python version, which you will find [here](https://pytorch.org/get-started/locally/#pip-1).

iNLTK runs on CPU, as is the desired behaviour for most
of the Deep Learning models in production.

The first command above will install pytorch for cpu, which, as the name suggests, does not have cuda support.

`Note: inltk is currently supported only on Linux and Windows 10 with Python >= 3.6`

### Supported languages

| Language | Code <code-of-language> |
|:--------:|:----:|
|   Hindi  |  hi  |
|  Punjabi |  pa  |
| Sanskrit |  sa  |
| Gujarati |  gu  |
|  Kannada |  kn  |
| Malayalam |  ml  |
|  Nepali  |  ne  |
|   Odia   |  or  |
|  Marathi |  mr  |
|  Bengali |  bn  |
|   Tamil  |  ta  |
|   Urdu  |  ur  |

### Usage

**Setup the language**

```bash
from inltk.inltk import setup

setup('<code-of-language>') // if you wanted to use hindi, then setup('hi')
```

`Note: You need to run setup('<code-of-language>') when you use a language 
for the FIRST TIME ONLY. This will download all the necessary models required
to do inference for that language.`

**Tokenize**

```bash
from inltk.inltk import tokenize

tokenize(text ,'<code-of-language>') // where text is string in <code-of-language>
```

**Get Embedding Vectors**

This returns an array of "Embedding vectors", containing 400 Dimensional representation for 
every token in the text.


```
from inltk.inltk import get_embedding_vectors

vectors = get_embedding_vectors(text, '<code-of-language>') // where text is string in <code-of-language>

Example:

>> vectors = get_embedding_vectors('भारत', 'hi')
>> vectors[0].shape
(400,)

>> get_embedding_vectors('ਜਿਹਨਾਂ ਤੋਂ ਧਾਤਵੀ ਅਲੌਹ ਦਾ ਆਰਥਕ','pa')
[array([-0.894777, -0.140635, -0.030086, -0.669998, ...,  0.859898,  1.940608,  0.09252 ,  1.043363], dtype=float32), array([ 0.290839,  1.459981, -0.582347,  0.27822 , ..., -0.736542, -0.259388,  0.086048,  0.736173], dtype=float32), array([ 0.069481, -0.069362,  0.17558 , -0.349333, ...,  0.390819,  0.117293, -0.194081,  2.492722], dtype=float32), array([-0.37837 , -0.549682, -0.497131,  0.161678, ...,  0.048844, -1.090546,  0.154555,  0.925028], dtype=float32), array([ 0.219287,  0.759776,  0.695487,  1.097593, ...,  0.016115, -0.81602 ,  0.333799,  1.162199], dtype=float32), array([-0.31529 , -0.281649, -0.207479,  0.177357, ...,  0.729619, -0.161499, -0.270225,  2.083801], dtype=float32), array([-0.501414,  1.337661, -0.405563,  0.733806, ..., -0.182045, -1.413752,  0.163339,  0.907111], dtype=float32), array([ 0.185258, -0.429729,  0.060273,  0.232177, ..., -0.537831, -0.51664 , -0.249798,  1.872428], dtype=float32)]
>> vectors = get_embedding_vectors('ਜਿਹਨਾਂ ਤੋਂ ਧਾਤਵੀ ਅਲੌਹ ਦਾ ਆਰਥਕ','pa')
>> len(vectors)
8

``` 

Links to `Embedding visualization` on [Embedding projector](https://projector.tensorflow.org/) for all the supported languages are given in table below. 

**Predict Next 'n' words**

```bash
from inltk.inltk import predict_next_words

predict_next_words(text , n, '<code-of-language>') 

// text --> string in <code-of-language>
// n --> number of words you want to predict (integer)
```

`Note: You can also pass a fourth parameter, randomness, to predict_next_words.
It has a default value of 0.8`

**Identify language**

Note: If you update the version of iNLTK, you need to run 
`reset_language_identifying_models` before identifying language.

```bash
from inltk.inltk import identify_language, reset_language_identifying_models

reset_language_identifying_models() # only if you've updated iNLTK version
identify_language(text)

// text --> string in one of the supported languages

Example:

>> identify_language('न्यायदर्शनम् भारतीयदर्शनेषु अन्यतमम्। वैदिकदर्शनेषु ')
'sanskrit'

```

**Remove foreign languages**

```bash
from inltk.inltk import remove_foreign_languages

remove_foreign_languages(text, '<code-of-language>')

// text --> string in one of the supported languages
// <code-of-language> --> code of that language whose words you want to retain

Example:

>> remove_foreign_languages('विकिपीडिया सभी विषयों ਇੱਕ ਅਲੌਕਿਕ ਨਜ਼ਾਰਾ ਬੱਝਾ ਹੋਇਆ ਸਾਹਮਣੇ ਆ ਖਲੋਂਦਾ ਸੀ पर प्रामाणिक और 维基百科:关于中文维基百科 उपयोग, परिवर्तन 维基百科:关于中文维基百科', 'hi')
['▁विकिपीडिया', '▁सभी', '▁विषयों', '▁', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁पर', '▁प्रामाणिक', '▁और', '▁', '<unk>', ':', '<unk>', '▁उपयोग', ',', '▁परिवर्तन', '▁', '<unk>', ':', '<unk>']
```

Every word other than that of host language will become `<unk>` and `▁` signifies `space character`

Checkout [this notebook](https://drive.google.com/file/d/0B3K0rqnCfC9pbVpSWk9Ndm5raGRCdjV6cGxVN1BGWFhTTlA0/view?usp=sharing)
 by [Amol Mahajan](https://www.linkedin.com/in/amolmahajan0804/) where he uses iNLTK to remove foreign characters from
 [iitb_en_hi_parallel corpus](http://www.cfilt.iitb.ac.in/iitb_parallel/iitb_corpus_download/)
 
 
 **Get Sentence Encoding**
 
```
from inltk.inltk import get_sentence_encoding

get_sentence_encoding(text, '<code-of-language>')

Example: 

>> encoding = get_sentence_encoding('मुझे अपने देश से', 'hi')
>> encoding.shape
(400,)
```

`get_sentence_encoding` returns 400 dimensional encoding of the sentence from
ULMFiT LM Encoder of `<code-of-language>` trained in repositories linked below.


**Get Sentence Similarity**

```
from inltk.inltk import get_sentence_similarity

get_sentence_similarity(sentence1, sentence2, '<code-of-language>', cmp = cos_sim)

// sentence1, sentence2 are strings in '<code-of-language>'
// similarity of encodings is calculated by using cmp function whose default is cosine similarity

Example: 

>> get_sentence_similarity('मैं इन दोनों श्रेणियों के बीच कुछ भी सामान्य नहीं देखता।', 'मैंने कन्फेक्शनरी स्टोर्स पर सेब और संतरे की कीमतों की तुलना की', 'hi')
0.126698300242424

>> get_sentence_similarity('मैं इन दोनों श्रेणियों के बीच कुछ भी सामान्य नहीं देखता।', 'यहां कोई तुलना नहीं है। आप सेब की तुलना संतरे से कर रहे हैं', 'hi')
0.25467658042907715
```

`get_sentence_similarity` returns similarity between two sentences by calculating
`cosine similarity` (default comparison function) between the encoding vectors of two
sentences.


**Get Similar Sentences**

```
from inltk.inltk import get_similar_sentences

get_similar_sentences(sentence, no_of_variants, '<code-of-language>')


Example:

>> get_similar_sentences('मैं आज बहुत खुश हूं', 10, 'hi')
['मैं आजकल बहुत खुश हूं',
 'मैं आज काफ़ी खुश हूं',
 'मैं आज काफी खुश हूं',
 'मैं अब बहुत खुश हूं',
 'मैं आज अत्यधिक खुश हूं',
 'मैं अभी बहुत खुश हूं',
 'मैं आज बहुत हाजिर हूं',
 'मैं वर्तमान बहुत खुश हूं',
 'मैं आज अत्यंत खुश हूं',
 'मैं सदैव बहुत खुश हूं']

```

`get_similar_sentences` returns `list` of length `no_of_variants` which contains sentences which
 are similar to `sentence`

#### Repositories containing models used in iNLTK
|  Language | Repository                                                       | Perplexity of Language model | Wikipedia Articles Dataset |   Classification accuracy   |     Classification Kappa score    |                                                     Embeddings visualization on [Embedding projector](https://projector.tensorflow.org/)                                                     |
|:---------:|------------------------------------------------------------------|:----------------------------:|:--------------------------:|:---------------------------:|:---------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   Hindi   | [NLP for Hindi](https://github.com/goru001/nlp-for-hindi)        |              ~36             |       55,000 articles      |  ~79 (News Classification)  | ~30 (Movie Review Classification) |  [Hindi Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-hindi/master/language-model/embedding_projector_config_30k.json)   |
|  Punjabi  | [NLP for Punjabi](https://github.com/goru001/nlp-for-punjabi)    |              ~13             |       44,000 articles      |  ~89 (News Classification)  |     ~60 (News Classification)     |   [Punjabi Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-punjabi/master/language-model/embedding_projector_config.json)  |
|  Sanskrit | [NLP for Sanskrit](https://github.com/goru001/nlp-for-sanskrit)  |              ~6              |       22,273 articles      | ~70 (Shloka Classification) |    ~56 (Shloka Classification)    |  [Sanskrit Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-sanskrit/master/language-model/embedding_projector_config.json) |
|  Gujarati | [NLP for Gujarati](https://github.com/goru001/nlp-for-gujarati)  |              ~34             |       31,913 articles      |  ~91 (News Classification)  |     ~85 (News Classification)     |  [Gujarati Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-gujarati/master/language-model/embedding_projector_config.json) |
|  Kannada  | [NLP for Kannada](https://github.com/goru001/nlp-for-kannada)    |              ~70             |       32,997 articles      |  ~94 (News Classification)  |     ~90 (News Classification)     |   [Kannada Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-kannada/master/language-model/embedding_projector_config.json)  |
| Malayalam | [NLP for Malayalam](https://github.com/goru001/nlp-for-malyalam) |              ~26             |       12,388 articles      |  ~94 (News Classification)  |     ~91 (News Classification)     | [Malayalam Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-malyalam/master/language-model/embedding_projector_config.json) |
|   Nepali  | [NLP for Nepali](https://github.com/goru001/nlp-for-nepali)      |              ~32             |       38,757 articles      |  ~97 (News Classification)  |     ~96 (News Classification)     |    [Nepali Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-nepali/master/language-model/embedding_projector_config.json)   |
|    Odia   | [NLP for Odia](https://github.com/goru001/nlp-for-odia)          |              ~27             |       17,781 articles      |  ~95 (News Classification)  |     ~92 (News Classification)     |      [Odia Embeddings Projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-odia/master/language-model/embedding_projector_config.json)     |
|  Marathi  | [NLP for Marathi](https://github.com/goru001/nlp-for-marathi)    |              ~18             |       85,537 articles      |  ~91 (News Classification)  |     ~84 (News Classification)     |   [Marathi Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-marathi/master/language-model/embedding_projector_config.json)  |
|  Bengali  | [NLP for Bengali](https://github.com/goru001/nlp-for-bengali)    |              ~41             |       72,374 articles      |  ~94 (News Classification)  |     ~92 (News Classification)     |   [Bengali Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-bengali/master/language-model/embedding_projector_config.json)  |
|   Tamil   | [NLP for Tamil](https://github.com/goru001/nlp-for-tamil)        |              ~20             |      >127,000 articles     |  ~97 (News Classification)  |     ~95 (News Classification)     |     [Tamil Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-tamil/master/language-model/embedding_projector_config.json)    |
|    Urdu   | [NLP for Urdu](https://github.com/anuragshas/nlp-for-urdu)       |              ~13             |      >150,000 articles     |  ~94 (News Classification)  |     ~90 (News Classification)     |    [Urdu Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/anuragshas/nlp-for-urdu/master/language-model/embedding_projector_config.json)    |

### Contributing

**Add a new language support for iNLTK**

If you would like to add support for language of your own choice to iNLTK,
 please start with checking/raising a issue [here](https://github.com/goru001/inltk/issues)
 
Please checkout the steps I'd [mentioned here for Telugu](https://github.com/goru001/inltk/issues/1)
to begin with. They should be almost similar for other languages as well.

**Improving models/Using models for your own research**

If you would like to take iNLTK's models and refine them with your own 
dataset or build your own custom models on top of it, please check out the 
repositories in the above table for the language of your choice. The repositories above 
contain links to datasets, pretrained models, classifiers and all of the code for that.

**Add new functionality**

If you wish for a particular functionality in iNLTK - Start by checking/raising a issue [here](https://github.com/goru001/inltk/issues)

### What's next (and being worked upon)

`Shout out if you want to help :)`

* Add [Telugu](https://github.com/goru001/inltk/issues/1) 
and [Maithili](https://github.com/goru001/inltk/issues/10) support
* Add NER support
* Add Textual Entailment support
* Add English to iNLTK


### What's next - (and NOT being worked upon)

`Shout out if you want to lead :)`

* Work on a [unified model for all the languages](https://github.com/goru001/inltk/issues/14)
* [POS support](https://github.com/goru001/inltk/issues/13) in iNLTK
* Add translations - to and from languages in iNLTK + English

### Appreciation for iNLTK 

* [By Jeremy Howard on Twitter](https://twitter.com/jeremyphoward/status/1111318198891110402)
* [By Vincent Boucher on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:6517137647310241792/)
* [By Kanimozhi](https://www.linkedin.com/feed/update/urn:li:activity:6517277916030701568), [By Soham](https://www.linkedin.com/feed/update/urn:li:activity:6513084638955696128), [By Imaad](https://www.linkedin.com/feed/update/urn:li:activity:6536258026687557632/) on LinkedIn
* iNLTK was [trending on GitHub](https://github.motakasoft.com/trending/ranking/monthly/?d=2019-05-01&l=python&page=2) in May 2019
* iNLTK has had [19,000+ Downloads](
https://console.cloud.google.com/bigquery?sq=375816891401:185fda81bdc64eb79b98c6b28c77a62a
) till Nov 2019


