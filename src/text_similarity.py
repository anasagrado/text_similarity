import sys
from nltk.corpus.reader import WordNetError
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.corpus import wordnet_ic
from gensim.models import Word2Vec
import gensim


class listUtils:

    def filter_None(x):
        return [w for w in x if w is not None]

class wordNet:
    """
    End2end pipeline to transform a sentence into a wordNet object sentence
    """
    def __init__(self, config = None):
        #super().__init__()
        self.config = config
        if config.get("lemmantize",False):
            self.lemmatizer = WordNetLemmatizer()


    def lemmatization(self, x):
        """

        :param x:
        :return:
        """
        return [self.lemmatizer.lemmatize(word) for word in x]

    def get_tag(self,x):
        """
        Given a list of words , this function returns the
        word with its associated tag
        :return: a list of tuples with the first element in
        the tuple the word and the second the tag
        """
        return nltk.pos_tag(x)

    def filter_tags(self,x, tags_filter = ["J","N","V","R"]):
        """
        By default the list of values to filter is ["J","N","V","R"] which correspond to  [adjetive, noun, verb and adverbs]
        If tags_filter is equal to "notIn" then it will retrieve everything that doesn't lie in the category of ["J","N","V","R"]
        This methods need to be used before copnverting into wordNet. The tags should be standard nltk tags
        Eg:  [('I', 'PRP'), ('go', 'VBP'), ('for', 'IN'), ('a', 'DT'), ('run', 'NN')] ->  [('go', 'VBP'), ('run', 'NN')]
        :param x:
        :return: the list filter by tag
        """
        if tags_filter == "notIn":
            return [(word,tag) for word, tag in x if tag[0] not in ["J","N","V","R"]]
        return [(word,tag) for word, tag in x if tag[0] in tags_filter]

    def get_the_first(self, x):
        """
        Given a list of list the method selects, for each list, the first element
        [[0,1],[2,3],[4,4]] -> [0,1,4]
        :param x: list of list
        :return: a list
        """
        return [y[0] for y in x if len(y) > 0]

    def get_wordnet_equiv(self,tag, default_value=None ):
        """
        :return: given a nltk tag returns the tag in the
        wordNet form. If the tag is not in the list it returns
        noun y default
        """
        tag = tag[0].upper()
        tag_dict = {"J": wn.ADJ,
                    "N": wn.NOUN,
                    "V": wn.VERB,
                    "R": wn.ADV}
        return tag_dict.get(tag, default_value)

    def get_wordnet_tag(self, x, default_value= None):
        """
        Given a list of tuples (word,tag)
        :return: a list of tuples (word,tag) in the
        wordNet form
        """
        return [(word, self.get_wordnet_equiv(tag, default_value)) for word, tag in x]

    def get_wordnet_lesk_synset(self, x):
        """
        Input is a list of tuples where the tuples should be( word, tag)
        :return: a list of sysnsets with the sense based on lesk algorithm
        """
        return [lesk(x, word, tag) for word, tag in x]

    def get_wordnet_and_word_lesk_synset(self, x):
        """
        Input is a list of tuples where the tuples should be( word, tag)
        :return: a list of sysnsets with the sense based on lesk algorithm
        """
        def syns_or_word(word,tag):

            word_syns = lesk(x, word, tag)
            #print(word, word_syns, type(word_syns))
            if type(word_syns) == nltk.corpus.reader.wordnet.Synset:
                result = word_syns
            elif word_syns is None:
                result = word
            else:
                raise ValueError("The word should be either a str or wordNet sysnset, but it's {} type ".format(type(word)))
            return result

        return [ syns_or_word(word, tag) for word, tag in x]


    def get_wordnet_value(self,x):
        """
        :param x: list of tuples, each tuple is (word,tag)
        :return: List with the correspondent wordNet
        """
        return [wn.synsets(word, pos=tag) for word, tag in x]

    def get_wordnet_num_syn(syn):
        """
        Given a wordSynset returns the number.
        This might be useful as the sysnets are order increasing based on the
        probability of find that word in a text
        Eg: get_num_syn(wn.synsets('bank')[0] ) -> 1
                 ---- where wn.synsets('bank')[0] = Synset('bank.n.01')
        :return:
        """
        return int(str(syn).split("(")[-1].split(")")[0].replace("'", "").split(".")[-1])

    def text_preparation(self, x):
        """
        This method creates the list of wordNets from a sentence

        :param x:
        :param config:
        :return:
        """

        if not self.config["isTokenized"]:
            x = word_tokenize(x)
        if self.config.get("lemmantize", False):
            x = self.lemmatization(x)
        x = self.get_tag(x)
        if self.config.get("filter_tags", None) is not None:
            ##filter tags
            x = self.filter_tags(x, self.config["filter_tags"] )
        if self.config["selection_method"] == "raw_words":

            # if the selection method is 'None' means to not apply the wordNet transformations
            # it will just keep the words
            return [word for word, tag in x ]
        x = self.get_wordnet_tag(x)

        if self.config["selection_method"] == "first_value":
            x = self.get_wordnet_value(x)
            x = self.get_the_first(x)
        elif self.config["selection_method"] == "lesk":
            x = self.get_wordnet_lesk_synset(x)
        elif self.config["selection_method"] == "word_and_lesk":
            x = self.get_wordnet_and_word_lesk_synset(x)
        else:
            raise ValueError("The selection method should be 'lesk' or 'first_value'or 'word_and_lesk or 'raw_words' ")
        return x


class textSimilarity(wordNet):
    """
    Class with methods for similarity.
    Needs a configuration which specifies the steps:
    - IC_type: Describe the data to use to create the information content
        --> Possible values: 'ic-brown.dat', 'ic-semcor.dat'
    - lexical_similarity: similarity metric to call
        --> Possible values: wordnet_path_similarity, wordnet_lch_similarity,
                             wordnet_wup_similarity, wordnet_res_similarity,
                             wordnet_jcn_similarity, wordnet_lin_similarity, similarity_word2vec
    - vector_similarity:
        --> Possible values: jaccard_similarity, cosine_similarity, mean_similarity
    - match_method:
        --> Possible values: similarity_match_double_mean, similarity_match_vector_method
    - similarity_threshold:
        --> Default values is 0 if other thing is not specify
    - vector_threshold
        --> if this values exists then filter all the vector based on this threshold
    """
    def __init__(self, config):
        super().__init__(config)
        self.__WordNetError__ = 0
        self.similarity_threshold = 0
        if config.get("IC_type", None) is not None:
            #corpus_ic = wordnet_ic.ic('ic-brown.dat')
            #corpus_ic = wordnet_ic.ic('ic-semcor.dat')
            self.corpus_ic = wordnet_ic.ic(config["IC_type"])
        if config.get("similarity_threshold", None) is not None:
            self.similarity_threshold = config["similarity_threshold"]
        try:
            # self.lexical_similarity = "wordnet_" + config["lexical_similarity"]
            self.lexical_similarity = config["lexical_similarity"]
            self.match_method = config.get("match_method","similarity_match_vector_method")
            self.vector_similarity = "v_" + config["vector_similarity"]
            #print("Using {} lexical similarity method  -- {} vector similarity method -- match method {} ".format(self.lexical_similarity, self.vector_similarity , self.match_method))
        except KeyError:
            print("Similarity_name property specify the type of similarity to apply.\n\
                           Currently implemented are: path_similarity, lch_similarity, wup_similarity, res_similarity, jcn_similarity, lin_similarity")
            pass

        # if self.lexical_similarity == "wordnet_" +"model_word2vec":
        if self.lexical_similarity == "similarity_word2vec" :
            self.model_word2vec = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin',
                                                                             binary=True)

        not_numerical_distances = ['v_jaccard_similarity']
        if self.vector_similarity in not_numerical_distances:
            self.numerical_representation = False
        else:
            self.numerical_representation = True

    ######################################################################
    # -------- methods for computing similarity between words -----------
    ######################################################################

    def wordnet_path_similarity(self, word1, word2):
        """
        between 0 as 1
        :param word2:
        :return:
        """
        return word1.path_similarity(word2)

    def wordnet_lch_similarity(self, word1, word2):
        """
        without upper bound
        :param word2:
        :return:
        """
        if word1.pos() != word2.pos():
            ##if the classes are completly different it means they are completly far
            return 0
        return word1.lch_similarity(word2)

    def wordnet_wup_similarity(self, word1, word2):
        """
        between 0 as 1
        :param word2:
        :return:
        """
        return word1.wup_similarity(word2)

    def wordnet_res_similarity(self, syns1, syns2):
        """
        IC based method. Need to have config["IC"] equal to True and
                     config["IC_type"] specifying the IC type
        :param syns2:
        :return:
        """
        if syns1.pos() != syns2.pos():
            return 0
        return syns1.res_similarity(syns2, self.corpus_ic)

    def wordnet_jcn_similarity(self, syns1, syns2):
        """
        IC based method. Need to have config["IC"] equal to True and
                     config["IC_type"] specifying the IC type
        :param syns2:
        :return:
        """
        if syns1.pos() != syns2.pos():
            return 0
        return syns1.jcn_similarity(syns2, self.corpus_ic)

    def wordnet_lin_similarity(self, syns1, syns2):
        """
        IC based method. Need to have config["IC"] equal to True and
                     config["IC_type"] specifying the IC type
        :param syns2:
        :return:
        """
        if syns1.pos() != syns2.pos():
            return 0
        return syns1.lin_similarity(syns2, self.corpus_ic)

    def similarity_word2vec(self, w1, w2):
        """
        Word2vect model is the only method to measure similarity between
        two words that is in this module
        :param w1: word1
        :param w2: word2
        :return: similarity between the two words
        """
        try:
            v1 = self.model_word2vec[w1]
            v2 = self.model_word2vec[w2]
            result = self.v_cosine_similarity(v1, v2)
        except KeyError:
            result = 0
        return result

    ######################################################################
    # -------- methods for computing similarity between words -----------
    ######################################################################

    def levenshtein_distance(self, word1, word2, normalized=True):
        m, n = len(word1), len(word2)
        DD = np.zeros((m + 1, n + 1))
        DD[:, 0] = range(m + 1)
        DD[0, :] = range(n + 1)

        for t1 in range(1, m + 1):
            for t2 in range(1, n + 1):
                if (word1[t1 - 1] == word2[t2 - 1]):
                    DD[t1][t2] = DD[t1 - 1][t2 - 1]
                else:
                    delete = DD[t1][t2 - 1]
                    insert = DD[t1 - 1][t2]
                    replace = DD[t1 - 1][t2 - 1]

                    cost_v = [replace, insert, delete]
                    min_cost = min(cost_v)

                    DD[t1][t2] = min_cost + 1
        distance = DD[-1][-1]
        if normalized:
            distance = distance / max(m, n)
        return distance


    def wordnet_and_match(self, similarity, word1, word2):
        """
        This function will try to get the similarity score from
        wordNet but if the words (word1 or word2) are not in
        the on the lexical database wordNet, then
        the similarity will be equal to one if the two words are equal

        """
        path_sim = None

        if type(word1) == nltk.corpus.reader.wordnet.Synset:
            if type(word2) == nltk.corpus.reader.wordnet.Synset:
                #path_sim = word1.path_similarity(word2)
                try:
                    path_sim = similarity(word1,word2)
                except WordNetError:
                    path_sim = None
                    self.__WordNetError__ += 1
                word1 = word1.lemma_names()[0]
                word2 = word2.lemma_names()[0]
                # print(word1,word2,path_sim)
                # if path_sim is not None and path_sim > 0.5:
                #     if word1 != word2:
                        #print(word1, word2)
            else:
                word1 = word1.lemma_names()[0] #convert the sysnset to str
        elif type( word2 ) == nltk.corpus.reader.wordnet.Synset: #implicit here that the word1 is a synset if we get into this if condition
            word2 = word2.lemma_names()[0]  # convert the sysnset to str
        word_sim = 1 if word1 == word2 else 0
        # word_sim = 1  if self.levenshtein_distance(word1, word2, normalized=False ) < 2 else 0
        path_sim = 0 if path_sim is None else path_sim
        return max(path_sim,word_sim)

    def wordnet_path_similarity_and_match(self, word1, word2):
        """
        Computes similarity between two words using
        path similarity
        """
        return self.wordnet_and_match(self.wordnet_path_similarity, word1, word2)

    def wordnet_lch_similarity_and_match(self, word1, word2):
        """
        Computes similarity between two words using
        lch similarity
        """
        return self.wordnet_and_match(self.wordnet_lch_similarity, word1, word2)

    def wordnet_wup_similarity_and_match(self, word1, word2):
        """
        Computes similarity between two words using
        wup similarity
        """
        return self.wordnet_and_match(self.wordnet_wup_similarity, word1, word2)

    def wordnet_res_similarity_and_match(self, word1, word2):
        """
        Computes similarity between two words using
        res similarity
        """
        return self.wordnet_and_match(self.wordnet_res_similarity, word1, word2)

    def wordnet_jcn_similarity_and_match(self, word1, word2):
        """
        Computes similarity between two words using
        jcn similarity. Note than to have a normalized value
        in here we need to divide up by 3e150.
        3e150 is the maximum value that jcn similarity can have
        """
        result = self.wordnet_and_match(self.wordnet_jcn_similarity, word1, word2)#/3e300
        return result/3e150

    def wordnet_just_similarity_and_match(self, word1, word2):
        """
        Simple similarity for exact match, i.e
        similarity is equal to 1 if words are equal, otherwise 0
        """
        res = 1 if word1 == word2 else 0
        return res

    #
    # def similarity_match_double_mean(self, row, col1, col2 ):
    #     """
    #     This method computes twice the mean for the similarity vectors
    #     For example:
    #     sim_vect_sentence1 = [ 0.001, 0.8973, 0.1232]
    #     sim_vect_sentence2 = [  0.8973 , 0]
    #     Then the similairyt between the two sentences will be
    #     mean( mean(sim_vect_sentence1),mean(sim_vect_sentence2) ) =
    #     mean( mean([ 0.001, 0.8973, 0.1232]),mean([  0.8973 , 0]) )  =
    #     mean( 0.3405 , 0.44865 ) = 0.394575
    #
    #     """
    #     threshold = self.similarity_threshold
    #     similarity = self.__getattribute__(self.lexical_similarity)
    #
    #     row1, row2 = row.get(col1), row.get(col2)
    #     row1 = listUtils.filter_None(row1)
    #     row2 = listUtils.filter_None(row2)
    #     sim1 = []
    #     sim2 = []
    #     if len(row1) == 0 or len(row2) == 0:
    #         return 0
    #     for syn1 in row1:
    #         # print(syn1, row1, row2)
    #         sym12 = listUtils.filter_None([similarity(syn1, syn2) for syn2 in row2])
    #         sym12 = [x if x > threshold else 0 for x in sym12]
    #         if sym12: sim1.append(np.max(sym12))
    #     for syn2 in row2:
    #         sym21 = listUtils.filter_None([similarity(syn1, syn2) for syn1 in row1])
    #         sym21 = [x if x > threshold else 0 for x in sym21]
    #         if sym21: sim2.append(np.max(sym21))
    #     # result = max(np.mean(sim1), np.mean(sim2))
    #     result = np.mean([np.mean(sim1), np.mean(sim2)])
    #     # result = np.mean([  max(np.mean(sim1), np.mean(sim2))  ,  np.mean([ np.mean(sim1), np.mean(sim2) ])   ])
    #     return result

    def similarity_numerical_vector(self,row, col1, col2):
        """
        This function will return the vector of similarities
        for two sentences.
        Eg:
        sent1 = ['I', 'like', 'walking']
        sent2 = ['I','like','walk','slowly']
        Then the output would look like
        [[1, 1, 0.8], [1,1,0.8, 0]]

        The output of this method will be used in one of the
        vector similarity v_* measures
        """
        threshold = self.similarity_threshold
        similarity = self.__getattribute__(self.lexical_similarity)
        row1, row2 = row.get(col1), row.get(col2)
        row1 = listUtils.filter_None(row1)
        row2 = listUtils.filter_None(row2)
        rows = None
        if len(row1) == 0 and len(row2) == 0:
            return [[0], [0]]
        elif len(row1) != 0 and len(row2) != 0:
            rows = row1 + row2
        elif len(row1) != 0:
            rows = row1
        elif len(row2) != 0:
            rows = row2
        bag_of_words = list(set(rows))
        # print(bag_of_words)
        sim1 = []
        sim2 = []
        for word in bag_of_words:
            sym1_ = listUtils.filter_None([similarity(syn1, word) for syn1 in row1])
            sym2_ = listUtils.filter_None([similarity(syn2, word) for syn2 in row2])
            # print(word,sim1, sym1_,sym2_)
            sym1_ = [x if x > threshold else 0 for x in sym1_]
            sym2_ = [x if x > threshold else 0 for x in sym2_]

            if len(sym1_) == 0:
                sym1_ = [0.0]
            if len(sym2_) == 0:
                sym2_ = [0.0]
            sim1.append(np.max(sym1_))
            sim2.append(np.max(sym2_))
        assert len(sim1) == len(sim2)
        return [sim1, sim2]


    def v_jaccard_similarity(self,v1, v2):
        """
        Jaccard similaruty between two vectors
        """
        s1 = set(v1)
        s2 = set(v2)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def v_cosine_similarity(self, v1, v2):


        if len(v1) == 0 or len(v2) == 0:
            return 0
        if np.sum(v1) == 0 or np.sum(v2) == 0:
            return 0
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def v_mean_similarity(self, v1, v2):
        """
        This method computes twice the mean for the similarity vectors
        For example:
        sim_vect_sentence1 = [ 0.001, 0.8973, 0.1232]
        sim_vect_sentence2 = [  0.8973 , 0]
        Then the similairyt between the two sentences will be
        mean( mean(sim_vect_sentence1),mean(sim_vect_sentence2) ) =
        mean( mean([ 0.001, 0.8973, 0.1232]),mean([  0.8973 , 0]) )  =
        mean( 0.3405 , 0.44865 ) = 0.394575
        """
        return np.mean( [np.mean(v1), np.mean(v2) ] )

    def similarity_match_vector_method(self,row, col1, col2 ):
        """
        Unless specify otherwise this method is the one using by default

        :param row: row with columns col1 and col2
        :param col1:
        :param col2:
        :return:
        """
        if self.numerical_representation:
            v1, v2 = self.similarity_numerical_vector(row, col1, col2 )
        else:
            v1,v2 = row.get(col1), row.get(col2)
        vector_similarity = self.__getattribute__(self.vector_similarity)
        if self.config.get("vector_threshold", False):
            vector_threshold = self.config["vector_threshold"]
            #print("Applying threshold {} to the vectors".format(vector_threshold))
            v1 = [ x if x > vector_threshold else 0 for x in v1 ]
            v2 = [ x if x > vector_threshold else 0 for x in v2 ]
        return vector_similarity(v1,v2)

    # def similarity_match_vector_mean(self,row, col1, col2 ):
    #     v1, v2 = self.similarity_match_vector(row, col1, col2 )
    #
    #     return max(np.mean(v1), np.mean(v2))

    def similarity_match_method(self,row, col1, col2):
        try:
            match_method_apply = self.__getattribute__(self.match_method)
        except AttributeError:
            AttributeError("{} is not currently implement. The methods implemented are similarity_match_vector_method and similarity_match_vector ".format(self.match_method))
        return match_method_apply(row, col1, col2)


if __name__ == "__main__":

    import pandas as pd

    train_path = '../data/sts-train.csv'
    train_path = '../data/sts-test.csv'
    df_train = pd.read_csv(train_path, sep='\t', error_bad_lines=False,
                           engine='python'
                           , names=["genre", "filename", "year", "score", "sentence1", "sentence2"])


    # def cleanText2(x):
    #     result = apply_re(x)
    #     result = toLowerCase(result)
    #     result = removeStopWords(result)
    #     result = lemmatization(result, False)
    #     return word_tokenize(result)
    #
    #
    # print("Missing {} rows, because there is nan values on them".format(
    #     len(df_train.index) - len(df_train.dropna().index)))
    # df_train = df_train.dropna()
    # df_train = df_train.reset_index()
    #
    # df_train["sentence1_frmt"] = df_train["sentence1"].apply(cleanText2)#.apply(lambda x: ' '.join(cleanText(x)) )
    # df_train["sentence2_frmt"] = df_train["sentence2"].apply(cleanText2)#.apply(lambda x: ' '.join(cleanText(x)) )

    config = {"isTokenized": False,
                "selection_method": "None",
                #                   "filter_tags" :  ["J","N","V","R"],
                "lexical_similarity": "model_word2vec",
                # "vector_similarity": "cosine_similarity",
                "match_method": "similarity_match_double_mean",
                #             "similarity_threshold" : 0.5,
                "vector_threshold": 0.7
                # "IC_type": 'ic-brown.dat'
                }

    __textSimilarity__ = textSimilarity(config)

    df_train["test"] = df_train[["sentence1","sentence2"]].applymap(__textSimilarity__.text_preparation).apply(
            lambda x: __textSimilarity__.similarity_match_method(
            x, "sentence1_frmt", "sentence2_frmt")
            , axis=1)


