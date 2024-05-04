from abc import abstractmethod
import json
import os
import typing
import pandas as pd
import phunspell
from tqdm import tqdm
import re

tqdm.pandas()


class DataFrameTransformer:
    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class ListTransformer(DataFrameTransformer):
    def __init__(self, processors: typing.List[DataFrameTransformer]):
        self._processors = processors

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for processor in self._processors:
            data = processor(data)
        return data


class SerbianCyrillicToLatinTransformer(DataFrameTransformer):
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        changes = {
            column: data[column].progress_apply(self._serbian_cyrillic_to_latin)
            for column in data.columns
        }
        return data.assign(**changes)

    def _serbian_cyrillic_to_latin(self, text: str) -> str:
        cyrillic_to_latin = {
            "а": "a",
            "б": "b",
            "в": "v",
            "г": "g",
            "д": "d",
            "ђ": "đ",
            "е": "e",
            "ж": "ž",
            "з": "z",
            "и": "i",
            "ј": "j",
            "к": "k",
            "л": "l",
            "љ": "lj",
            "м": "m",
            "н": "n",
            "њ": "nj",
            "о": "o",
            "п": "p",
            "р": "r",
            "с": "s",
            "т": "t",
            "ћ": "ć",
            "у": "u",
            "ф": "f",
            "х": "h",
            "ц": "c",
            "ч": "č",
            "џ": "dž",
            "ш": "š",
            "А": "A",
            "Б": "B",
            "В": "V",
            "Г": "G",
            "Д": "D",
            "Ђ": "Đ",
            "Е": "E",
            "Ж": "Ž",
            "З": "Z",
            "И": "I",
            "Ј": "J",
            "К": "K",
            "Л": "L",
            "Љ": "Lj",
            "М": "M",
            "Н": "N",
            "Њ": "Nj",
            "О": "O",
            "П": "P",
            "Р": "R",
            "С": "S",
            "Т": "T",
            "Ћ": "Ć",
            "У": "U",
            "Ф": "F",
            "Х": "H",
            "Ц": "C",
            "Ч": "Č",
            "Џ": "Dž",
            "Ш": "Š",
        }

        # Replace Cyrillic characters with Latin characters
        latin_text = "".join(cyrillic_to_latin.get(char, char) for char in text)
        return latin_text


class SpellCheckTransformer(DataFrameTransformer):
    ERROR = "__ERROR__"

    def __init__(self, language_code: str = "sr"):
        self._pspell = phunspell.Phunspell(language_code)

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        changes = {
            column: data[column].progress_apply(self._correct)
            for column in data.columns
        }
        for k, v in changes.items():
            mask = v == SpellCheckTransformer.ERROR
            print(f"{k} has {mask.sum()} out of {len(mask)} errors.")
            changes[k] = v.mask(mask, data[k])
        return data.assign(**changes)

    def _correct(self, text: str) -> str:
        try:
            return next(self._pspell.suggest(text))
        except StopIteration as e:
            return SpellCheckTransformer.ERROR


class CapitalizeTransformer(DataFrameTransformer):
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        changes = {
            column: data[column].progress_apply(self._correct)
            for column in data.columns
        }
        return data.assign(**changes)

    @staticmethod
    def _correct(sentence: str):
        corrected_sentence = sentence.capitalize()

        # Capitalize the first letter of each sentence
        sentences = corrected_sentence.split(". ")
        corrected_sentences = [sentence.capitalize() for sentence in sentences]

        # Join the corrected sentences back together
        return ". ".join(corrected_sentences)


class WordOccurence(DataFrameTransformer):
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        ocurences = self._ocurences(data)
        print(
            json.dumps(
                dict(sorted(ocurences.items(), key=lambda item: item[1], reverse=True)),
                indent=4,
                ensure_ascii=False,
            )
        )
        return data

    @staticmethod
    def _ocurences(data: pd.DataFrame):
        series_concatenated = pd.Series(data.values.flatten())

        # Initialize an empty dictionary to store word counts
        word_count = {}

        # Iterate over each sentence in the concatenated Series
        for sentence in series_concatenated:
            # Split the sentence into words
            words = sentence.split()
            # Count the occurrences of each word
            for word in words:
                # Increment the count for the word or initialize it to 1 if it doesn't exist
                word_count[word] = word_count.get(word, 0) + 1

        return word_count


class Column(DataFrameTransformer):
    def __init__(self, processor: DataFrameTransformer, columns: list):
        self._columns = columns
        self._processor = processor

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        changes = {col: self._processor(data[[col]]) for col in self._columns}
        return data.assign(**changes)


class Replacement(DataFrameTransformer):
    def __init__(self, replacements: dict):
        self._replacements = replacements

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        changes = {
            column: data[column].progress_apply(self._correct)
            for column in data.columns
        }
        return data.assign(**changes)

    def _correct(self, text: str) -> str:
        for k, v in self._replacements.items():
            text = text.replace(k, v)

        return text


general_transforms = [SerbianCyrillicToLatinTransformer(), WordOccurence()]
ANSWER = "answer"
QUESTION = "question"
processors = {
    "potera.csv": ListTransformer(
        general_transforms
        + [
            Column(Replacement({x + " ": "" for x in "ABV"}), [ANSWER]),
            Column(CapitalizeTransformer(), [ANSWER]),
        ]
    ),
    "slagalica.csv": ListTransformer(
        [CapitalizeTransformer(), Replacement({'"': ""})] + general_transforms
    ),
}
data_path = "data"
output_data_path = "data_out"
for data_file, processor in processors.items():
    data = pd.read_csv(os.path.join(data_path, data_file)).fillna("")
    data[[QUESTION, ANSWER]] = processor(data[[QUESTION, ANSWER]])
    os.makedirs(output_data_path, exist_ok=True)
    output_data_file = os.path.join(output_data_path, data_file)
    data.to_csv(output_data_file, index=False)
