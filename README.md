# word2vec_modelling
Find projection between word2vec models trained on different datasets

## Run

Code is available on [Google colab](https://colab.research.google.com/github/evjeny/word2vec_modelling/blob/main/runner.ipynb)

*Note: set accelerator device to GPU: **Menu -> Runtime -> Change runtime -> Hardware acceleration = GPU***

## Scripts

### [split_texts.py](split_texts.py)

Script downloads texts from the **20 newsgroups** dataset,
filters words and numbers and then saves files to directory

### [generate_folds.py](generate_folds.py)

Script splits documents from previous step into subsets with determined document overlap

Note: document overlap is calculated between two adjacent document subsets, not between each subset with each subset

For example, if we had subsets: `[subset1, subset2, subset3, subset4]` and overlap `30%`, that would mean that `subset1 and subset2`, `subset2 and subset3`, `subset3 and subset4` have `30%` of same documents.

### [train_word2vec.py](train_word2vec.py)

Script trains Word2Vec model on each subset of documents

### [train_mapping.py](train_mapping.py)

Script trains mapping between each embedding space of given Word2Vec models

### [circle_mapping.py](circle_mapping.py)

Script measures word similarity of given Word2Vec embedding with embedding outputed by several mappings between other Word2Vec embedding spaces

For example, we can get embedding from model `Word2Vec1`, then map it to embedding space of model `Word2Vec2`, then map result to embedding space of model `Word2Vec3` and then map result to `Word2Vec1` embedding space again, and after all calculate cosine similarity between source and transformed embeddings

So overall scheme will be like:

1. word -> `Word2Vec1_embedding`
2. `Word2Vec1_embedding` -> `Word2Vec2_embedding`
3. `Word2Vec2_embedding` -> `Word2Vec3_embedding`
4. `Word2Vec3_embedding` -> `Word2Vec1_embedding_new`
5. `cosine_similarity(Word2Vec1_embedding, Word2Vec1_embedding_new)`

### [domain_mapping.py](domain_mapping.py)

Script calculate word embeddings of the same word for two models, then maps embedding of the first model to embedding space of the second model and measures cosine similarity

Shortly, script does:

1. word -> `Word2Vec1_embedding`
2. word -> `Word2Vec2_embedding`
3. `Word2Vec1_embedding` -> `Word2Vec2_embedding_new`
4. `cosine_similarity(Word2Vec2_embedding_new, Word2Vec2_embedding)`
