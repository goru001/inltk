## Natural Language Toolkit for Indic Languages (iNLTK)

[![Gitter](https://badges.gitter.im/inltk/community.svg)](https://gitter.im/inltk/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Downloads](https://pepy.tech/badge/inltk)](https://pepy.tech/project/inltk)

iNLTK aims to provide out of the box support for various NLP tasks 
that an application developer might need for Indic languages.


### Documentation

Checkout detailed docs along with Installation instructions
 at https://inltk.readthedocs.io


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
* [By Sebastian Ruder on Twitter](https://twitter.com/seb_ruder/status/1207074241830674438)
* [By Vincent Boucher on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:6517137647310241792/)
* [By Kanimozhi](https://www.linkedin.com/feed/update/urn:li:activity:6517277916030701568), [By Soham](https://www.linkedin.com/feed/update/urn:li:activity:6513084638955696128), [By Imaad](https://www.linkedin.com/feed/update/urn:li:activity:6536258026687557632/) on LinkedIn
* iNLTK was [trending on GitHub](https://github.motakasoft.com/trending/ranking/monthly/?d=2019-05-01&l=python&page=2) in May 2019
* iNLTK has had [23,000+ Downloads](
https://console.cloud.google.com/bigquery?sq=375816891401:185fda81bdc64eb79b98c6b28c77a62a
) on PyPi
