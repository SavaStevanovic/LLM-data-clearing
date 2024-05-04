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


class WordOccurence(DataFrameTransformer):

    def __init__(self, fixes: list = None):
        fixes = fixes if fixes else []
        self._fixes = fixes

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        ocurences = {
            k: v for k, v in self._ocurences(data).items() if k not in self._fixes
        }
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


mapping = {
    "Godine": "godine",
    "rosija,": "Rosija,",
    "valentin": "Valentin",
    "banjaj": "Banjaj",
    "dbpxxkabe": "države",
    "rahmon": "Rahmon",
    "stiven": "Stiven",
    "oliver": "Oliver",
    "injca": "Injca",
    "banjaj": "Banjaj",
    "„": '"',
    "“": '"',
    "alžir": "Alžir",
    '""': '"',
    "jusef": "Jusef",
    "hatavej": "Hatavej",
    "hale": "Hale",
    "fajfer": "Fajfer",
    "hatavej": "Hatavej",
    "nikola": "Nikola",
    "eurosong": "Eurosong",
    "stefan": "Stefan",
    "dragan": "Dragan",
    "jovanović": "Jovanović",
    "aleksandar": "Aleksandar",
    "uefa": "UEFA",
    "jugoslavije": "Jugoslavije",
    "vladimir": "Vladimir",
    "africi": "Africi",
    "americi": "Americi",
    "nemačkoj": "Nemačkoj",
    "italiji": "Italiji",
    "popović": "Popović",
    "evropi": "Evropi",
    "amerike": "Amerike",
    "njujorku": "Njujorku",
    "isidora": "Isidora",
    "pansa": "Pansa",
    "zzimmskimm": "zimskim",
    "dijego": "Dijego",
    "odanović": "Odanović",
    "olgom": "Olgom",
    "piterson": "Piterson",
    "anđeles": "Anđeles",
    "kerber": "Kerber",
    "malik": "Malik",
    "monro": "Monro",
    "Bu ": "",
    "beogradu": "Beogradu",
    "venecueli": "Venecueli",
    "kertis": "Kertis",
    "branka": "Branka",
    "ćopića": "Ćopića",
    "skorceni": "Skorceni",
    "holandiji": "Holandiji",
    "Maks plank": "Maks Plank",
    "Kil bil": "Kil Bil",
    "petković dis": "Petković Dis",
    "petković": "Petković",
    "manro": "Manro",
    "Stivi vonder": "Stivi Vonder",
    "skandinaviji": "Skandinaviji",
    "prelević": "Prelević",
    "ljubek": "Ljubek",
    "vitman": "Vitman",
    "pirsu": "Pirsu",
    "popović": "Popović",
    "krunić": "Krunić",
    "saverin": "Saverin",
    "mjanmaru": "Mjanmaru",
    "malden": "Malden",
    "hornasek": "Hornasek",
    "timora": "Timora",
    "gotjeu": "Gotjeu",
    "ivanović": "Ivanović",
    "australiji": "Australiji",
    "švedskoj": "Švedskoj",
    "bolpačić": "Bolpačić",
    "Mona Liz": "Mona Liz",
    "patrik": "Patrik",
    "Haris": "Haris",
    "kanada": "Kanada",
    "de žaneir": "De Žaneir",
    "eliot": "Eliot",
    "afrike": "Afrike",
    "andrića": "Andrića",
    ' "': '"',
    " ?": "?",
    " .": ".",
    " !": "!",
    '" ': '"',
    "rohas": "Rohas",
    "najrobiju": "Najrobiju",
    "kristi": "Kristi",
    "pelagić": "Pelagić",
    "marija": "Marija",
    "hauer": "Hauer",
    "anketil": "Anketil",
    "rajh": "Rajh",
    "wilhelm": "Wilhelm",
    "reich": "Reich",
    "nele karajlić": "Nele Karajlić",
    "karajlić": "Karajlić",
    "iranu": "Iranu",
    "kolumbije": "Kolumbije",
    "regan": "Regan",
    "malteze": "Malteze",
    "haksli": "Haksli",
    "martina": "Martina",
    "miloš": "Miloš",
    "jovića": "Jovića",
    "delon": "Delon",
    "holandije": "Holandije",
    "dubrovnik": "Dubrovnik",
    "lajović": "Lajović",
    "kapone": "Kapone",
    "belgiji": "Belgiji",
    "todorović": "Todorović",
    "da gama": "da Gama",
    "marino": "Marino",
    "bogart": "Bogart",
    "ćopić": "Ćopić",
    "lorens": "Lorens",
    "nušić": "Nušić",
    "montano": "Montano",
    "raspućin": "Raspućin",
    "arabija": "Arabija",
    "spasić": "Spasić",
    "mekre": "Mekre",
    "bata živojinović": "Bata Živojinović",
    "živojinović": "Živojinović",
    "džonson": "Džonson",
    "Loh nes": "Loh Nes",
    "baltimoru": "Baltimoru",
    "obradović": "Obradović",
    "vasiljev": "Vasiljev",
    "de sika": "de Sika",
    "šekularac": "Šekularac",
    "ramacoti": "Ramacoti",
    "petrović": "Petrović",
    "njegoš": "Njegoš",
    '"baz"': '"Baz"',
    "stanković": "Stanković",
    "nišavi": "Nišavi",
    "japanu": "Japanu",
    "leonov": "Leonov",
    "real madrid": "Real Madrid",
    "madrid": "Madrid",
    "berlinu": "Berlinu",
    "sudan": "Sudan",
    "meksiku": "Meksiku",
    "gojković": "Gojković",
    "cune gojković": "Cune Gojković",
    "luksemburg": "Luksemburg",
    "savić": "Savić",
    "makijaveli": "Makijaveli",
    "indije": "Indije",
    "mionici": "Mionici",
    "šubašić": "Šubašić",
    "indonezije": "Indonezije",
    "korać": "Korać",
    "bojanić": "Bojanić",
    "gidra": "Gidra",
    "stokholmu": "Stokholmu",
    "bičer stou": "Bičer Stou",
    "šekularac": "Šekularac",
    "australije": "Australije",
    "finskoj": "Finskoj",
    "gandolfini": "Gandolfini",
    "jorović": "Jorović",
    "bulatović": "Bulatović",
    "robert": "Robert",
    "huku": "Huku",
    "stenli": "Stenli",
    "metjuz": "Metjuz",
    "garašanin": "Garašanin",
    "savić": "Savić",
    "uelbek": "Uelbek",
    "milutinović": "Milutinović",
    "mika": "Mika",
    "antić": "Antić",
    "hoking": "Hoking",
    "sparou": "Sparou",
    "bogović": "Bogović",
    "aranđelovc": "Aranđelovc",
    "rumunije": "Rumunije",
    "canić": "Canić",
    "šumanović": "Šumanović",
    "aronofski": "Aronofski",
    "landštajner": "Landštajner",
    "veličković": "Veličković",
    "haneke": "Haneke",
    "loren": "Loren",
    "bunjuel": "Bunjuel",
    "o'nil": "O'nil",
    "harison": "Harison",
    "šekspir": "Šekspir",
    "kurdistana": "Kurdistana",
    "dikens": "Dikens",
    "karenjina": "Karenjina",
    "simović": "Simović",
    "karađorđević": "Karađorđević",
    "Valentino rosi": "Valentino Rosi",
    "džordž": "Džordž",
    "ferari": "Ferari",
    "džinović": "Džinović",
    "jakšić": "Jakšić",
    "konan": "Konan",
    "dojl": "Dojl",
    "belgije": "Belgije",
    "andresku": "Andresku",
    "montija": "Montija",
    "montija pajtona": "Montija Pajtona",
    "dravić": "Dravić",
    "šupljikac": "Šupljikac",
    "bekuta": "Bekuta",
    "Leni kravic": "Leni Kravic",
    "nadarević": "Nadarević",
    "domanović": "Domanović",
    "mihail": "Mihail",
    "obrenovića": "Obrenovića",
    "medvedev": "Medvedev",
    "gustav": "Gustav",
    "jung": "Jung",
    "antonije": "Antonije",
    "bursać": "Bursać",
    "jagodini": "Jagodini",
    "zaječaru": "Zaječaru",
    "veneciji": "Veneciji",
    "bogdanović": "Bogdanović",
    "zemunu": "Zemunu",
    "topalović": "Topalović",
    "puškin": "Puškin",
    "azije": "Azije",
    "velaskez": "Velaskez",
    "pekić": "Pekić",
    "gvineja": "Gvineja",
    "tokiju": "Tokiju",
    "labović": "Labović",
    "veletanlić": "Veletanlić",
    "lampard": "Lampard",
    "načić": "Načić",
    "amadeus": "Amadeus",
    "mocart": "Mocart",
    "angelopulos": "Angelopulos",
    "bugarskoj": "Bugarskoj",
}
general_transforms = [
    SerbianCyrillicToLatinTransformer(),
    Replacement(mapping),
    WordOccurence(list(mapping.values())),
]
ANSWER = "answer"
QUESTION = "question"
processors = {
    "potera.csv": ListTransformer(
        general_transforms
        + [
            Column(Replacement({x + " ": "" for x in "ABVD5"}), [ANSWER]),
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
