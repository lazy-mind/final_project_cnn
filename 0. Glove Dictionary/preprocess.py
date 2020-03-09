#!/usr/bin/env python
"""NLP Preprocessing Library"""
import nltk
import zipfile
import os
import json
import re
import string

class Preprocessor():

    def __init__(self, dict_path="dict/glove_index.txt", max_length_tweet=30, max_length_dictionary=124688):
        module_path = os.path.abspath(__file__)
        dict_path = os.path.join(os.path.dirname(module_path), dict_path)

        self.dict_path = dict_path
        self.max_length_tweet = max_length_tweet
        self.max_length_dictionary = max_length_dictionary
        self.mode = ""
        self.corpus = [""]*self.max_length_dictionary
        self.load_corpus()

    def log_variable(self):
        print(f"the dictionary path is {self.dict_path}")
        print(f"the max length of tweet is {self.max_length_tweet}")
        print(f"the max length of dictionary is {self.max_length_dictionary}")
        print(f"the mode is {self.mode}")

    def load_corpus(self):
        if ".zip/" in self.dict_path:
            split = self.dict_path.split(".zip/")
            archive_path = split[0] + ".zip"
            path_inside = split[1]

            archive = zipfile.ZipFile(archive_path, "r")
            embeddings = archive.read(path_inside).decode("utf8").split("\n")
            self.corpus = self.process_dictionary(embeddings)
            self.mode = "zip"
        else:
            file = open(self.dict_path, "r", encoding='utf-8')
            idx = 0
            for word in file:
                self.corpus[idx] = word.rstrip()
                idx += 1
            self.corpus = self.process_dictionary(self.corpus)
            self.mode = ""

    def process_dictionary(self, _list):
        """remove unknown or unk tag, insert pad and unknown tag"""
        try:
            del _list[_list.index('<unknown>')]
        except ValueError:
            pass

        try:
            del _list[_list.index('<unk>')]
        except ValueError:
            pass

        try:
            del _list[_list.index('<pad>')]
        except ValueError:
            pass

        _list.insert(0, '<unknown>')
        _list.insert(0, '<pad>')

        return _list

    def preprocess_text(self, input_text):
        """a general method to call, convert string to vectorized representation"""
        input_text = self.clean_text(input_text)
        tokenization_list = self.tokenize_text(input_text)
        index_list = self.replace_token_with_index(tokenization_list, self.max_length_dictionary)
        index_list = self.pad_sequence(index_list, self.max_length_tweet)
        return index_list

    def remove_punctuation(self, char):
        if char=="'" or char==";":
            return ""
        elif char in string.punctuation:
            return " "
        else:
            return char

    def clean_text(self, raw_text):
        """Remove url, tokens"""
        raw_text = raw_text.replace('RT ', '')
        text_list = raw_text.split()
        useless_tokens = ["@", "#"]
        text_list = [x for x in text_list if x[0] not in useless_tokens]
        text_list = [x for x in text_list if x.lower().find('http://') == -1]
        text_list = [x for x in text_list if x.lower().find('https://') == -1]
        text = (" ".join(text_list)).lower()
        text = re.sub(r'\d+', '', text)
        text = "".join([self.remove_punctuation(char) for char in text])
        # remove other tokens
        return text

    def tokenize_text(self, tweet_str):
        """convert string to chunks of text"""
        tokenizer = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
        return tokenizer.tokenize(tweet_str)
        # Source: https://www.nltk.org/api/nltk.tokenize.html

    def replace_token_with_index(self, tokenized_tweet, max_length_dictionary):
        """convert each text to dictionary index"""

        # should be replacing each token in a list of tokens by their corresponding index
        # Source: https://github.com/stanfordnlp/GloVe
        # Source: https://towardsdatascience.com/word-embeddings-for-sentiment-analysis-65f42ea5d26e

        for idx, word in enumerate(tokenized_tweet):
            if word not in self.corpus:
                tokenized_tweet[idx] = "<unknown>"
        return [self.corpus.index(x) for x in tokenized_tweet]

    def pad_sequence(self, arr, max_length_tweet):
        """add 0 padding to the trail until max_length_tweet"""
        # padding a list of indices with 0 until a maximum length (max_length_tweet)
        if max_length_tweet > len(arr):
            trailing_zeros = [0]*(max_length_tweet-len(arr))
            arr.extend(trailing_zeros)
        return arr[:max_length_tweet]
