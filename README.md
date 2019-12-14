## Natural Language Toolkit for Indic Languages (iNLTK)

[![Gitter](https://badges.gitter.im/inltk/community.svg)](https://gitter.im/inltk/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

iNLTK aims to provide out of the box support for various NLP tasks 
that an application developer might need for Indic languages.

![Alt Text](inltk/static/inltk.gif)

### Documentation

Checkout detailed docs at https://inltk.readthedocs.io


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

##### Add a new language support

If you would like to add support for language of your own choice to iNLTK,
 please start with checking/raising a issue [here](https://github.com/goru001/inltk/issues)
 
Please checkout the steps I'd [mentioned here for Telugu](https://github.com/goru001/inltk/issues/1)
to begin with. They should be almost similar for other languages as well.

##### Improving models/using models for your own research

If you would like to take iNLTK's models and refine them with your own 
dataset or build your own custom models on top of it, please check out the 
repositories in the above table for the language of your choice. The repositories above 
contain links to datasets, pretrained models, classifiers and all of the code for that.

##### Add new functionality

If you wish for a particular functionality in iNLTK - Start by checking/raising a issue [here](https://github.com/goru001/inltk/issues)


### What's next


#### ..and being worked upon
`Shout out if you want to help :)`

* Add [Telugu](https://github.com/goru001/inltk/issues/1) 
and [Maithili](https://github.com/goru001/inltk/issues/10) support
* Add NER support
* Add Textual Entailment support
* Add English to iNLTK


#### ..and NOT being worked upon

`Shout out if you want to lead :)`

* Work on a [unified model for all the languages](https://github.com/goru001/inltk/issues/14)
* [POS support](https://github.com/goru001/inltk/issues/13) in iNLTK
* Add translations - to and from languages in iNLTK + English



### iNLTK's Appreciation

* [By Jeremy Howard on Twitter](https://twitter.com/jeremyphoward/status/1111318198891110402)
* [By Vincent Boucher on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:6517137647310241792/)
* [By Kanimozhi](https://www.linkedin.com/feed/update/urn:li:activity:6517277916030701568), [By Soham](https://www.linkedin.com/feed/update/urn:li:activity:6513084638955696128), [By Imaad](https://www.linkedin.com/feed/update/urn:li:activity:6536258026687557632/) on LinkedIn
* iNLTK was [trending on GitHub](https://github.motakasoft.com/trending/ranking/monthly/?d=2019-05-01&l=python&page=2) in May 2019
* iNLTK has had [20,000+ Downloads](
https://console.cloud.google.com/bigquery?sq=375816891401:185fda81bdc64eb79b98c6b28c77a62a
) on PyPi
