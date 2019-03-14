import en_core_web_sm
import textacy
import pickle
import enchant
from collections import OrderedDict
from nltk.corpus import stopwords



class ParaphraseGenerationModelPhraseStringMatch():

    def __init__(self):

        self._ppdb_dataset = self._get_ppdb_dataset('examples/ppdb_dict.pkl')
        self._nlp = en_core_web_sm.load()
        self._list_of_stopwords = set(stopwords.words('english'))
        self.language_dictionary = enchant.Dict('en_US')


    def _get_ppdb_dataset(self, ppdb_dict_pkl_file=""):

        ppdb_dict = pickle.load(open(ppdb_dict_pkl_file, 'rb'))[0]

        return ppdb_dict

    def _extract_verb_phrase_from_sentence(self, sentence):
        """
        This function will extract verb phrases from the input sentences and will return the list of
        verb phrases.
        :param sentence: input sentence (String)
        :return: list of verb phrases in sentence (list of string)\
        """

        verb_clause_pattern = r'<VERB>*<ADV>*<PART>*<VERB>+<PART>*'
        doc = textacy.Doc(sentence, lang='en_core_web_sm')
        lists = textacy.extract.pos_regex_matches(doc, verb_clause_pattern)
        list_of_verb_phrases = []
        for list in lists:
            if str(list.text).islower():
                list_of_verb_phrases.append(list.text)

        return list_of_verb_phrases

    def _extract_noun_phrase_from_sentence(self, sentence):
        """
        This function will extract verb phrases from the input sentences and will return the list of
        verb phrases.
        :param sentence: input sentence (String)
        :return: list of verb phrases in sentence (list of string)
        """

        noun_clause_pattern = textacy.constants.POS_REGEX_PATTERNS['en']['NP']
        doc = textacy.Doc(sentence, lang='en_core_web_sm')
        lists = textacy.extract.pos_regex_matches(doc, noun_clause_pattern)
        list_of_noun_phrases = []
        for list in lists:
            list_of_noun_phrases.append(list.text)

        return list_of_noun_phrases

    def _get_phrase_replacement_from_ppdb_dataset(self, phrase):
        """
        This function returns the list of paraphrases of verb phrase from ppdb dataset.
        :param phrase: any phrase like noun phrase, verb phrase, adjective etc. (String)
        :return: list of paraphrases from ppdb dataset (list of String)
        """

        if str(phrase) in self._ppdb_dataset:
            possible_replacements = self._ppdb_dataset[str(phrase)]
            return possible_replacements
        else:
            if len(self._ppdb_dataset) == 0:
                print('WARNING: ppdb dict is empty')
            return []

    def generate_paraphrases_with_verb_replacement(self, sentence):
        """
        This function generate paraphrases by replacing verb phrases and ver in a sentence.
        :param sentence: text (string)
        :return: dictionary with paraphrases and score (dict)
        """

        preprocessed_sentence = sentence
        dict_of_paraphrases_with_scores = {}

        list_of_verb_phrases = self._extract_verb_phrase_from_sentence(preprocessed_sentence)
        for verb_phrase in list_of_verb_phrases:
            list_of_verb_phrase_split = str(verb_phrase).lower().split()
            if len([i for i in list_of_verb_phrase_split if i in self._list_of_stopwords]) == 0:
                possible_replacements = self._get_phrase_replacement_from_ppdb_dataset(verb_phrase)
                for replacement in possible_replacements:
                    # Please note that here we will replace all the occurrence of verb phrase with new verb phrase.
                    # First check whether new suggestion is actual word in english or not.
                    if self.language_dictionary.check(str(replacement['sentence_2'])):
                        generated_paraphrase = str(
                            preprocessed_sentence.replace(" " + str(verb_phrase).strip() + " ",
                                                          " " + str(
                                                              replacement['sentence_2']) + " "))

                        # Check whether the new verb phrase is considered as a verb phrase in new generated paraphrase.
                        list_of_verb_phrases_in_generated_paraphrase = self._extract_verb_phrase_from_sentence(
                            generated_paraphrase)
                        if str(replacement['sentence_2']) in list_of_verb_phrases_in_generated_paraphrase:
                            dict_of_paraphrases_with_scores[generated_paraphrase] = float(replacement['ppdb_score2'])

        return dict_of_paraphrases_with_scores


    def generate_paraphrases_with_noun_replacement(self, sentence):
        """
        This function generate paraphrases by replacing noun phrases
        :param sentence: text (string)
        :return: dictionary with paraphrases and score (dict)
        """

        preprocessed_sentence = sentence
        dict_of_paraphrases_with_scores = {}

        list_of_noun_phrases = self._extract_noun_phrase_from_sentence(preprocessed_sentence)
        for noun_phrase in list_of_noun_phrases:
            possible_replacements = self._get_phrase_replacement_from_ppdb_dataset(noun_phrase)
            for replacement in possible_replacements:
                # Mote that here we will replace all the occurrence of noun phrase with new verb phrase.
                # First check whether new suggestion is actual word in english or not.
                if self.language_dictionary.check(str(replacement['sentence_2'])):
                    generated_paraphrase = str(
                        preprocessed_sentence.replace(" " + str(noun_phrase).strip() + " ",
                                                      " " + str(
                                                          replacement['sentence_2']) + " "))

                    # Check whether the new verb phrase is considered as a verb phrase in new generated paraphrase.
                    list_of_noun_phrases_in_generated_paraphrase = self._extract_noun_phrase_from_sentence(
                        generated_paraphrase)
                    if str(replacement['sentence_2']) in list_of_noun_phrases_in_generated_paraphrase:
                        dict_of_paraphrases_with_scores[generated_paraphrase] = float(replacement['ppdb_score2'])

        return dict_of_paraphrases_with_scores

    def generate_paraphrases_with_n_gram_replacement(self, sentence):
        """
        This function generate paraphrases by replacing 1,2,3- gram in a sentence.
        :param sentence: text (string)
        :return: dictionary with paraphrases and score (dict)
        """

        dict_of_paraphrases_with_scores = {}
        preprocessed_sentence = sentence
        doc = textacy.Doc(preprocessed_sentence, lang='en_core_web_sm')
        list_of_ngram = []
        list_of_unigram = list(textacy.extract.ngrams(doc, 1, filter_stops=True, filter_punct=True, filter_nums=False))
        list_of_bigram = list(textacy.extract.ngrams(doc, 2, filter_stops=True, filter_punct=True, filter_nums=False))
        list_of_trigram = list(textacy.extract.ngrams(doc, 3, filter_stops=True, filter_punct=True, filter_nums=False))

        list_of_ngram.extend(list_of_unigram)
        list_of_ngram.extend(list_of_bigram)
        list_of_ngram.extend(list_of_trigram)

        for n_gram in list_of_ngram:
            possible_replacements = self._get_phrase_replacement_from_ppdb_dataset(n_gram)
            for replacement in possible_replacements:
                if self.language_dictionary.check(str(replacement['sentence_2'])):
                    generated_paraphrase = str(
                        preprocessed_sentence.replace(" " + str(n_gram).strip() + " ",
                                                      " " + str(
                                                          replacement['sentence_2']) + " "))
                    dict_of_paraphrases_with_scores[generated_paraphrase] = float(replacement['ppdb_score2'])

        return dict_of_paraphrases_with_scores


    def generate_paraphrases(self, sentence, max_number_of_paraphrases=1, verb_phrase_replacement=True,
                             noun_phrase_replacement=False, n_gram_replacement=False):
        """
        This function will generate paraphrase of given input sentence.

        :param sentence: sentence (String)
        :param max_number_of_paraphrases: maximum number of paraphrases of sentence required (Integer)
        :param verb_phrase_replacement: if enabled verb phrase replacement will be searched in PPDB dict for generating
        paraphrases (bool)
        :param noun_phrase_replacement: if enabled noun phrase replacement will be searched in PPDB dict for generating
        paraphrases (bool)
        :param n_gram_replacement: if enabled (1,2,3)-gram replacement will be searched in PPDB dict for generating
        paraphrases (bool)
        :return: if model is able to generate paraphrase for input sentence, then
        list of top n paraphrases.(List of String)

        Note: confidence is the weighted sum of score we got from bigram language model and ppdb score for replacement.
        """

        # First preprocess input sentence
        # preprocessed_sentence = self._preprocess_sentence(sentence)
        preprocessed_sentence = sentence
        if not preprocessed_sentence == "":
            dict_of_paraphrases_with_scores = {}

            # Firstly do it for verb phrase
            if verb_phrase_replacement:
                paraphrases_dict_with_verb_replacements = self.generate_paraphrases_with_verb_replacement(preprocessed_sentence)
                dict_of_paraphrases_with_scores.update(paraphrases_dict_with_verb_replacements)

            if noun_phrase_replacement:
                paraphrases_dict_with_noun_replacements = self.generate_paraphrases_with_noun_replacement(
                    preprocessed_sentence)
                dict_of_paraphrases_with_scores.update(paraphrases_dict_with_noun_replacements)

            if n_gram_replacement:
                paraphrases_dict_with_n_gram_replacements = self.generate_paraphrases_with_n_gram_replacement(
                    preprocessed_sentence)
                dict_of_paraphrases_with_scores.update(paraphrases_dict_with_n_gram_replacements)

            dict_of_paraphrases_with_scores = OrderedDict(
                sorted(dict_of_paraphrases_with_scores.items(), key=lambda x: float(x[1]), reverse=True))

            all_list_of_paraphrases = list(dict_of_paraphrases_with_scores.keys())
            if len(all_list_of_paraphrases) > max_number_of_paraphrases:
                return all_list_of_paraphrases[0:max_number_of_paraphrases]
            else:
                return all_list_of_paraphrases
        else:
            return []
