import re
import sys
import evaluate
import unicodedata
import pandas as pd

wer = evaluate.load("wer")
cer = evaluate.load("cer")

def strip_diacritics(text):
    normalized = unicodedata.normalize('NFD', text)
    stripped = ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize('NFC', stripped)


punctuations = ''.join([chr(i) for i in list(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))])

def remove_punctuation(word):
    return word.translate(str.maketrans('', '', re.sub('[@% ]','', punctuations))).lower()


class EvaluateModel():
    def __init__(self, model_name, reference, prediction):
        self.model_name = model_name
        self.reference = [remove_punctuation(text) for text in reference]
        self.prediction = [remove_punctuation(text) for text in prediction] 
        self.reference_wo_diacritics = [strip_diacritics(text) for text in self.reference]
        self.prediction_wo_diacritics = [strip_diacritics(text) for text in self.prediction]

    def wer_(self):
        return 100 * wer.compute(references=self.reference, predictions=self.prediction)

    def cer_(self):
        return 100 * cer.compute(references=self.reference, predictions=self.prediction)

    def eval_hallucination_in_prediction(self):
        _wer = 100 * wer.compute(references=self.reference_wo_diacritics, predictions=self.prediction_wo_diacritics)
        _cer = 100 * cer.compute(references=self.reference_wo_diacritics, predictions=self.prediction_wo_diacritics)
        return _wer, _cer
    
    def evaluate(self):
        wer = self.wer_()
        cer = self.cer_()
        hallucination_wer, hallucination_cer = self.eval_hallucination_in_prediction()
        return {
            'model_name': self.model_name,
            'wer': wer,
            'cer': cer,
            'hallucination_wer': hallucination_wer,
            'hallucination_cer': hallucination_cer
        }