import importlib.metadata
import inspect
import re
from typing import Any, Type, cast

import malaya
import numpy as np
import torch
from malaya.dictionary import is_english as MalayaIsEnglish
from malaya.text.function import PUNCTUATION as MalayaPUNCTUATION
from malaya.text.function import case_of as MalayaCaseOf
from malaya.text.function import is_emoji as MalayaIsEmoji
from malaya.text.regex import _date as MalayaDate
from malaya.text.regex import _expressions as MalayaExpressions
from malaya.text.regex import _money as MalayaMoney
from malaya.torch_model.huggingface import Tagging
from malaya.torch_model.huggingface import Tagging as MalayaTagging
from malaya.torch_model.rnn import Stem as MalayaStem
from malaya_boilerplate.torch_utils import to_numpy
from transformers import PreTrainedTokenizerFast

__version__ = importlib.metadata.version("malaya_tagging_pipeline")
WHITESPACE_SPLITTER_RE = re.compile(r"\s+")


def custom_malayaload(
    model: str,
    class_model: Type[Tagging],
    available_huggingface: dict[str, Any],
    force_check: bool = True,
    path: str = __name__,
    **kwargs: Any
    ) -> Tagging:
    additional_parameters = {
        'from_lang': 'from lang',
        'to_lang': 'to lang',
    }
    if model not in available_huggingface and force_check:
        raise ValueError(
            f'model not supported, please check supported models from `{path}.available_huggingface`.'
        )

    args = inspect.getfullargspec(class_model) #  args = inspect.getargspec(class_model)
    for k, v in additional_parameters.items():
        if k in args.args:
            kwargs[k] = available_huggingface[model].get(v)

    return class_model(
        model=model,
        **kwargs,
    )

malaya.pos.load = custom_malayaload  # ty: ignore[invalid-assignment]


def stem_tokens(lemmatizer: MalayaStem, tokens: list[str]) -> list[str]:
    lemmatizer.model.eval()
    with torch.inference_mode(mode=True):
        lemmas: list[str] = []
        for no, word in enumerate(tokens):
            if word in MalayaPUNCTUATION:
                lemmas.append(word)
            elif (
                re.findall(MalayaMoney, word.lower())
                or re.findall(MalayaDate, word.lower())
                or re.findall(MalayaExpressions['email'], word.lower())
                or re.findall(MalayaExpressions['url'], word.lower())
                or re.findall(MalayaExpressions['hashtag'], word.lower())
                or re.findall(MalayaExpressions['phone'], word.lower())
                or re.findall(MalayaExpressions['money'], word.lower())
                or re.findall(MalayaExpressions['date'], word.lower())
                or re.findall(MalayaExpressions['time'], word.lower())
                or re.findall(MalayaExpressions['ic'], word.lower())
                or re.findall(MalayaExpressions['user'], word.lower())
                or MalayaIsEmoji(word.lower())
                or MalayaIsEnglish(word.lower())
            ):
                lemmas.append(word)
            else:
                lemmas.append(MalayaCaseOf(word)(lemmatizer.stem_word(word)))
        if len(lemmas) != len(tokens):
            raise ValueError("The number of lemmas is not the same as the number of "
                            f"tokens: {len(lemmas)} != {len(tokens)}")
        return lemmas

def tag_tokens(pos_tagger: MalayaTagging, tokens: list[str]) -> list[str]:
    pos_tagger.model.eval()
    with torch.inference_mode(mode=True):
        pos_tokenizer = pos_tagger.tokenizer
        pos_tokenizer = cast(PreTrainedTokenizerFast, pos_tokenizer)
        tokenized_inputs = pos_tokenizer([tokens], truncation=True, is_split_into_words=True)

        label_ids = []
        word_ids: list[int] = tokenized_inputs.word_ids(batch_index=0)
        # Used to determine at decoding time the first sub-word token in a word/token.
        for _ in tokens:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(1)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

        padded = tokenized_inputs

        # converts to torch tensors
        for k in padded.keys():
            padded[k] = torch.from_numpy(np.array(padded[k])).to(pos_tagger.model.device)

        pred = pos_tagger.model(**padded)[0]
        predictions = to_numpy(pred)[0].argmax(axis=1)
        pos_tags: list[str] = []
        for i in range(len(predictions)):
            if label_ids[i] == -100:
                continue
            token = tokens[word_ids[i]]
            if token in MalayaPUNCTUATION:
                pos_tags.append("PUNCT")
                continue
            pos_tags.append(pos_tagger.rev_vocab[int(predictions[i])])
        if len(pos_tags) != len(tokens):
            raise ValueError("The number of tags is not the same as the number of "
                            f"tokens: {len(pos_tags)} != {len(tokens)}")
        return pos_tags


def word_tokenize(tokenizer, text: str, lowercase: bool) -> list[str]:
    tokens = tokenizer.tokenize(text, lowercase=lowercase)
    split_tokens = []
    for token in tokens:
        split_tokens.extend(WHITESPACE_SPLITTER_RE.split(token))
    return split_tokens


