import json
import logging
import re
import string

import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob

import numpy as np
import torch
import transformers

class Passage_Relevance_Model():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def find_relevance(self, context, query, passage):
        print('We have to enter into PRM module')
        inputs = self.tokenizer(context + query + "\nPassage: " + passage, return_tensors="pt").to('cuda')
        generate_ids = self.model.generate(**inputs, max_length=512, temperature=0)
        generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
        result = self.tokenizer.decode(generate_ids)
        print('Result from PRM is ',result)
        if result == "relevance":
            return True
        elif result == "irrelevance":
            return False
        else:
            print(f"Invalid output on PRM query: {context + query}")
            return False
