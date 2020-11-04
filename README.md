Text similarity
---------
This module aims to provide an easy way to test diferent text similarity techniques. 
Altought there is more powerfull techniques nowadays,  approximations based on wordNet 
and word2vect are still used.

For evaluating the different methods we used the STS (Semantic Textual Similarity) dataset 
used in the SemEval task. We measure the effectiveness of each method with the correlation
of the given similarity and the one calculated by our module

As the general rule word2vect has shown better results that wordNet approaches.


Usage
---------

textSimilarity class should be instanced with a configuration in a dictionary form
that specify the diferent steps to perform to the similarity between two classes


PARAMETERS NEEDED
- is_tokenized :  specify weather the input sentence is tokenized or not <br />
- selection_method : method to select the concept in wordNet <br />
    a. 'lesk' : use lesk algorithm to decide which concept to choose <br />
    b. 'first_value' : choose the first value among all the concepts <br />
    c. 'word_and_lesk : it'll use the lesk algorithm to choose the concept
        but in case the word is not in wordNet db, it will preserve the word
        as a string <br />
    d. 'raw_words' : use this method when using the lexical similarity functions
        similarity_word2vec and words_similarity <br />
- lexical_similarity : similarity measure between two word, Currently implemented <br />
    a. wordnet_path_similarity <br />
    b. wordnet_lch_similarity <br />
    c. wordnet_wup_similarity <br />
    d. wordnet_res_similarity : needs corpusIC <br />
    e. wordnet_jcn_similarity : needs corpusIC <br />
    f. wordnet_lin_similarity : needs corpusIC <br />
    g. words_similarity <br />
    h. similarity_word2vec <br />
- vector_similarity : given two vectors this function will give a similarity score <br />
    a. jaccard_similarity : this function differs from the other vector function
       because it works on the raw word and doesn't need a lexical similarity function <br />
    b. cosine_similarity : this function will use the cosine_similairty to compute the
       distance between two numerical vectors  <br />
    c. mean_similarity : this function will compute the double mean of the vectors  <br />

OPTIONAL PARAMETERS
- corpus_ic: Describe the data to use to create the information content  <br />
    a. 'ic-brown.dat'  <br />
    b. 'ic-semcor.dat'  <br />

Other preprocessing optional parameters are  <br />
- is_tokenized : Weather the input sentence is tokenized or not :  True or False  <br />
- lemmantize: If True it will lemmantize the sentence  <br />
- filter_tags : list of tags to filter among ["J","N","V","R"]  <br />

- vector_threshold : numerical lower threshold to decide weather or not
  to account for that similarity. For examples, if we have the similarity vectors  <br />
  v1 = [0.01, 0.5, 0.8] and v2 = [ 0.0, 0.06, 0.9]  <br />
  and we set the threshold to 0.1 then v1 and v2 will become  <br />
  v1 = [0.0 , 0.5, 0.8] and v2 = [ 0.0, 0.0, 0.9]


Once you have create your config dictionary

```python
config = {"is_tokenized": False,
           "selection_method": "lesk",
           "lexical_similarity": "wordnet_lin_similarity",
           "vector_similarity": "cosine_similarity", 
           "IC_type": 'ic-brown.dat'
           }
```
You can create an instance of the class
```python
text_similarity_instance = textSimilarity(config)
```
From here the easiest way to use this module is calling the apply method of the class.

Suppose that we have a pandas dataframe with two columns 'sentences1' and 'sentence2' and the want 
to know how similart they are


You can create an instance of the class
```python
df.apply(lambda x : text_similarity_instance.apply(
            x, "sentence1", "sentence2") , axis = 1 )
```

