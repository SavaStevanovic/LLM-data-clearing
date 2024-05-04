from abc import abstractmethod
import os
import typing
import pandas as pd

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
        changes = {column: data[column].apply(self._serbian_cyrillic_to_latin) for column in data.columns}
        return data.assign(**changes)

    def _serbian_cyrillic_to_latin(self, text: str) -> str:
        cyrillic_to_latin = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 'е': 'e', 'ж': 'ž', 'з': 'z',
            'и': 'i', 'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj', 'м': 'm', 'н': 'n', 'њ': 'nj', 'о': 'o',
            'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c',
            'ч': 'č', 'џ': 'dž', 'ш': 'š',
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Ђ': 'Đ', 'Е': 'E', 'Ж': 'Ž', 'З': 'Z',
            'И': 'I', 'Ј': 'J', 'К': 'K', 'Л': 'L', 'Љ': 'Lj', 'М': 'M', 'Н': 'N', 'Њ': 'Nj', 'О': 'O',
            'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'Ћ': 'Ć', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C',
            'Ч': 'Č', 'Џ': 'Dž', 'Ш': 'Š'
        }
        
        # Replace Cyrillic characters with Latin characters
        latin_text = ''.join(cyrillic_to_latin.get(char, char) for char in text)
        return latin_text


processor = ListTransformer([SerbianCyrillicToLatinTransformer()])
data_path = "data"
output_data_path = "data_out"
for data_file in os.listdir(data_path):
    data = pd.read_csv(os.path.join(data_path, data_file)).fillna("")
    data[["question", "answer"]] = processor(data[["question", "answer"]])
    os.makedirs(output_data_path, exist_ok=True)
    output_data_file = os.path.join(output_data_path, data_file)
    data.to_csv(output_data_file)