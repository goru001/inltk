### Installation

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

### Supported Languages

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

### API

#### Setup the language

```bash
from inltk.inltk import setup

setup('<code-of-language>') // if you wanted to use hindi, then setup('hi')
```

`Note: You need to run setup('<code-of-language>') when you use a language 
for the FIRST TIME ONLY. This will download all the necessary models required
to do inference for that language.`

#### Tokenize

```bash
from inltk.inltk import tokenize

tokenize(text ,'<code-of-language>') // where text is string in <code-of-language>
```

#### Get Embedding Vectors

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

#### Predict Next 'n' words

```bash
from inltk.inltk import predict_next_words

predict_next_words(text , n, '<code-of-language>') 

// text --> string in <code-of-language>
// n --> number of words you want to predict (integer)
```

`Note: You can also pass a fourth parameter, randomness, to predict_next_words.
It has a default value of 0.8`

#### Identify language

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

#### Remove foreign languages

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
 
 
#### Get Sentence Encoding
 
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


#### Get Sentence Similarity

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


#### Get Similar Sentences

```
from inltk.inltk import get_similar_sentences

get_similar_sentences(sentence, no_of_variants, '<code-of-language>', degree_of_aug = 0.1)

// where degree_of_aug is roughly the percentage of sentence you want to augment, with a default value of 0.1

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






### Trained Models

|  Language |                            Repository                            |                                                                                     Dataset used for Language modeling                                                                                     | Perplexity of ULMFiT LM | Perplexity of TransformerXL LM |                                                                                  Dataset used for Classification                                                                                 | Classification Accuracy | Classification Kappa score |                                                                                ULMFiT Embeddings visualization                                                                               |                                                                                  TransformerXL Embeddings visualization                                                                                  |
|:---------:|:----------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------:|:------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------:|:--------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   Hindi   |     [NLP for Hindi](https://github.com/goru001/nlp-for-hindi)    | [Hindi Wikipedia Articles - 172k](https://www.kaggle.com/disisbig/hindi-wikipedia-articles-172k)<br><br><br>[Hindi Wikipedia Articles - 55k](https://www.kaggle.com/disisbig/hindi-wikipedia-articles-55k) |  34.06<br><br><br>35.87 |     26.09<br><br><br>34.78     | [Hindi Movie Reviews Dataset](https://www.kaggle.com/disisbig/hindi-movie-reviews-dataset)<br><br><br>[BBC Hindi News Dataset](https://github.com/NirantK/hindi2vec/releases/tag/bbc-hindi-v0.1) |  61.66<br><br><br>79.79 |   42.29<br><br><br>73.01   |   [Hindi Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-hindi/master/language-model/embedding_projector_config_30k.json)  |    [Hindi Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-hindi/master/language-model/embedding_projector_config_transformerxl.json)   |
|  Punjabi  |   [NLP for Punjabi](https://github.com/goru001/nlp-for-punjabi)  |                                                          [Punjabi Wikipedia Articles](https://www.kaggle.com/disisbig/punjabi-wikipedia-articles)                                                          |          24.40          |              14.03             |                                                           [Punjabi News Dataset](https://www.kaggle.com/disisbig/punjabi-news-dataset)                                                           |          89.17          |            54.5            |   [Punjabi Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-punjabi/master/language-model/embedding_projector_config.json)  |   [Punjabi Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-punjabi/master/language-model/embedding_projector_transformer_config.json)  |
|  Sanskrit |  [NLP for Sanskrit](https://github.com/goru001/nlp-for-sanskrit) |                                                         [Sanskrit Wikipedia Articles](https://www.kaggle.com/disisbig/sanskrit-wikipedia-articles)                                                         |            ~6           |               ~3               |                                                       [Sanskrit Shlokas Dataset](https://www.kaggle.com/disisbig/sanskrit-shlokas-dataset)                                                       |           84.3          |            76.1            |  [Sanskrit Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-sanskrit/master/language-model/embedding_projector_config.json) |  [Sanskrit Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-sanskrit/master/language-model/embedding_projector_transformer_config.json) |
|  Gujarati |  [NLP for Gujarati](https://github.com/goru001/nlp-for-gujarati) |                                                         [Gujarati Wikipedia Articles](https://www.kaggle.com/disisbig/gujarati-wikipedia-articles)                                                         |          34.12          |              28.12             |                                                          [Gujarati News Dataset](https://www.kaggle.com/disisbig/gujarati-news-dataset)                                                          |           92.4          |            87.9            |  [Gujarati Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-gujarati/master/language-model/embedding_projector_config.json) |  [Gujarati Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-gujarati/master/language-model/embedding_projector_transformer_config.json) |
|  Kannada  |   [NLP for Kannada](https://github.com/goru001/nlp-for-kannada)  |                                                          [Kannada Wikipedia Articles](https://www.kaggle.com/disisbig/kannada-wikipedia-articles)                                                          |          70.10          |              61.97             |                                                           [Kannada News Dataset](https://www.kaggle.com/disisbig/kannada-news-dataset)                                                           |           95.9          |            93.04           |   [Kannada Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-kannada/master/language-model/embedding_projector_config.json)  |   [Kannada Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-kannada/master/language-model/embedding_projector_transformer_config.json)  |
| Malayalam | [NLP for Malayalam](https://github.com/goru001/nlp-for-malyalam) |                                                        [Malayalam Wikipedia Articles](https://www.kaggle.com/disisbig/malayalam-wikipedia-articles)                                                        |          26.39          |              25.79             |                                                          [Malayalam News Dataset](https://www.kaggle.com/disisbig/malyalam-news-dataset)                                                         |          94.36          |            91.54           | [Malayalam Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-malyalam/master/language-model/embedding_projector_config.json) | [Malayalam Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-malyalam/master/language-model/embedding_projector_transformer_config.json) |
|   Nepali  |    [NLP for Nepali](https://github.com/goru001/nlp-for-nepali)   |                                                           [Nepali Wikipedia Articles](https://www.kaggle.com/disisbig/nepali-wikipedia-articles)                                                           |           31.5          |              29.3              |                                                            [Nepali News Dataset](https://www.kaggle.com/disisbig/nepali-news-dataset)                                                            |           98.5          |            97.7            |    [Nepali Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-nepali/master/language-model/embedding_projector_config.json)   |    [Nepali Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-nepali/master/language-model/embedding_projector_transformer_config.json)   |
|    Odia   |      [NLP for Odia](https://github.com/goru001/nlp-for-odia)     |                                                             [Odia Wikipedia Articles](https://www.kaggle.com/disisbig/odia-wikipedia-articles)                                                             |          26.57          |              26.81             |                                                              [Odia News Dataset](https://www.kaggle.com/disisbig/odia-news-dataset)                                                              |          95.52          |            93.02           |      [Odia Embeddings Projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-odia/master/language-model/embedding_projector_config.json)     |      [Odia Embeddings Projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-odia/master/language-model/embedding_projector_transformer_config.json)     |
|  Marathi  |   [NLP for Marathi](https://github.com/goru001/nlp-for-marathi)  |                                                          [Marathi Wikipedia Articles](https://www.kaggle.com/disisbig/marathi-wikipedia-articles)                                                          |            18           |              17.42             |                                                           [Marathi News Dataset](https://www.kaggle.com/disisbig/marathi-news-dataset)                                                           |          93.55          |            87.50           |   [Marathi Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-marathi/master/language-model/embedding_projector_config.json)  |   [Marathi Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-marathi/master/language-model/embedding_projector_transformer_config.json)  |
|  Bengali  |   [NLP for Bengali](https://github.com/goru001/nlp-for-bengali)  |                                                          [Bengali Wikipedia Articles](https://www.kaggle.com/disisbig/bengali-wikipedia-articles)                                                          |           41.2          |              39.3              |                                                           [Bengali News Dataset](https://www.kaggle.com/disisbig/bengali-news-dataset)                                                           |           93.8          |             92             |   [Bengali Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-bengali/master/language-model/embedding_projector_config.json)  |   [Bengali Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-bengali/master/language-model/embedding_projector_transformer_config.json)  |
|   Tamil   |     [NLP for Tamil](https://github.com/goru001/nlp-for-tamil)    |                                                            [Tamil Wikipedia Articles](https://www.kaggle.com/disisbig/tamil-wikipedia-articles)                                                            |          19.80          |              17.22             |                                                             [Tamil News Dataset](https://www.kaggle.com/disisbig/tamil-news-dataset)                                                             |          96.78          |            95.09           |     [Tamil Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-tamil/master/language-model/embedding_projector_config.json)    |     [Tamil Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-tamil/master/language-model/embedding_projector_transformer_config.json)    |
|    Urdu   |    [NLP for Urdu](https://github.com/anuragshas/nlp-for-urdu)    |                                                             [Urdu Wikipedia Articles](https://www.kaggle.com/disisbig/urdu-wikipedia-articles)                                                             |          13.19          |              12.55             |                                                              [Urdu News Dataset](https://www.kaggle.com/disisbig/urdu-news-dataset)                                                              |          95.28          |            91.58           |    [Urdu Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/anuragshas/nlp-for-urdu/master/language-model/embedding_projector_config.json)    |    [Urdu Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/anuragshas/nlp-for-urdu/master/language-model/embedding_projector_transformer_config.json)    |



