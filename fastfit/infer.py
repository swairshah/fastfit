import logging
import os
import random
import sys
import json
import math
import tempfile
import uuid

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Optional

import torch
import datasets
import numpy as np
from datasets import load_dataset, load_metric
import argparse

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    pipeline,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from transformers.integrations import INTEGRATION_TO_CALLBACK
from .modeling import ConfigArguments
from .modeling import FastFitTrainable, FastFit, FastFitConfig

def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

def get_top_k_predictions(text, k):
    input_tokens = tokenizer(text, return_tensors="pt")
    #model_outputs = model(**input_tokens)
    model_outputs = model(input_tokens['input_ids'], input_tokens['attention_mask'])
    scores = softmax(model_outputs["logits"][0].numpy())
    top_k_scores = np.argsort(scores)[-k:][::-1]
    predictions = [(model.config.id2label[score], scores[score].item()) for score in top_k_scores]
    return predictions

def calculate_accuracy(inputs, ground_truths, top_k=None):
    correct_predictions = 0
    total_predictions = len(inputs)

    for text, true_label in tqdm(zip(inputs, ground_truths)):
        input_tokens = tokenizer(text, return_tensors="pt")
        #model_outputs = model(**input_tokens)
        model_outputs = model(input_tokens['input_ids'], input_tokens['attention_mask'])
        scores = softmax(model_outputs["logits"][0].numpy())
        top_predictions = np.argsort(scores)[-top_k:][::-1] if top_k else [scores.argmax().item()]

        if true_label in [model.config.id2label[pred] for pred in top_predictions]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

def find_mistakes(inputs, ground_truths, top_k):
    mistakes = {"text": [] , "label": [], "candidates" : []}

    for text, true_label in tqdm(zip(inputs, ground_truths)):
        input_tokens = tokenizer(text, return_tensors="pt")
        model_outputs = model(input_tokens['input_ids'], input_tokens['attention_mask'])
        scores = softmax(model_outputs["logits"][0].numpy())
        top_k_scores = np.argsort(scores)[-top_k:][::-1]
        top_k_labels = [model.config.id2label[score] for score in top_k_scores]
           
        if true_label not in top_k_labels:
            mistakes['text'].append(text)
            mistakes['label'].append(true_label)
            mistakes['candidates'].append(top_k_labels)
    return mistakes

#text = "CUSTOMER: did't get my order"
#preds = get_top_k_predictions(text, 3)
#print(preds)

#input_tokens = tokenizer(text, return_tensors="pt")
#model_outputs = model(**input_tokens)
#scores = sigmoid(model_outputs["logits"][0])
#print(model.config.id2label[scores.argmax().item()])
#print(scores.max().item())

#classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
#classifier(text)

#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#model = FastFit.from_pretrained("i3/all-MiniLM-L6-v2/checkpoint-175/")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
model = FastFit.from_pretrained("tmp/paraphrase-mpnet-base-v2/checkpoint-175/")

import pandas as pd
test_data = "../data/explanation_dataset_test.json"
df = pd.read_json(test_data, lines=True)
texts, labels = df['text'].values.tolist(), df['label'].values.tolist()
acc = calculate_accuracy(texts, labels, top_k=3)
print(acc)

mistakes = find_mistakes(texts, labels, 3)
mistakes = pd.DataFrame(mistakes)

import IPython; IPython.embed()
