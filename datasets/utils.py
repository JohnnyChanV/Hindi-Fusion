

sentence = "ऴक़ख़"
phonetic_replace = {'क़': 'क़', 'ख़': 'ख़', 'ग़': 'ग़', 'ज़': 'ज़', 'ड़': 'ड़', 'ढ़': 'ढ़', 'फ़': 'फ़', 'ऩ': 'ऩ', 'ऱ': 'ऱ', 'ऴ': 'ऴ'}

vowel = {'ा': 'ā', 'ॅ': 'æ', 'ॉ': 'ɒ', 'ि': 'i', 'ी': 'ī', 'ु': 'u', 'ू': 'ū', 'ॆ': 'e', 'े': 'ē', 'ै': 'ai',
         'ॊ': 'o', 'ो': 'ō', 'ौ': 'au', 'ृ': 'r̥', 'ॄ': 'r̥̄', 'ॢ': 'l̥', 'ॣ': 'l̥̄', 'ं': 'aṁ', 'ः': 'aḥ', '्': '',
         '़': 'e', 'ँ': 'h'}

single_vowel = {'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī', 'उ': 'u', 'ऊ': 'ū', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
                'अं': 'aṁ', 'अः': 'aḥ'}


consonant = {'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ṅ', 'च': 'c', 'छ': 'ch', 'ज': 'j', 'झ': 'jh', 'ञ': 'ñ',
             'ट': 'ṭ', 'ठ': 'ṭh', 'ड': 'ḍ', 'ढ': 'ḍh', 'ण': 'ṇ', 'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
             'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm', 'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'ś',
             'ष': 'ṣ', 'स': 's', 'ह': 'h'}

punctuation = {'०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
               '।': '.'}

from wxconv import WXC
import re

class LatinConvert(object):
    def __init__(self, low_case=False):
        self.low_case = low_case
        self.con = WXC(order="utf2wx")

    def _convert_one(self, sentence):
        if self.low_case:
            return self.con.convert(sentence.lower()).replace("A", "aa").replace("N", "n").replace("D", "dh") .replace("I", "ii")\
    .replace("U", "uu").replace("M", "n").replace("E", "ai").replace("w", "t").replace("P", "ph").replace("x", "d")\
    .replace("G", "gh").replace("K", "kh").replace("S", "sh").replace("O", "au").replace("X", "dh") .replace("W", "th")\
    .replace("B", "bh").replace("dZ", "dr").replace("R", "shh").replace("Y", "").replace("J", "jh").replace("C", "ch")\
    .replace("T", "thh").replace("F", "gn").replace("DZ", "dr").replace("jZ", "z").replace("hZ", "h").replace("kZ", "k")\
    .replace("gZ", "g")
        else:
            return self.con.convert(sentence)

    def convert_many(self, sentences):
        return [self._convert_one(e) for e in sentences]

    def convert(self, sentence):
        return self._convert_one(sentence)

    def convert_ignore_upper(self, sentence):
        return " ".join([self._convert_one(e.lower()) if not re.match(r".*[a-zA-Z]+.*", e) else e for e in sentence.split()])


def devanagari_to_latin(string):
    con = LatinConvert()
    res = con.convert_ignore_upper(string)
    return res


def is_devanagari(string):
    for i in string:
        if 2304 <= ord(i) <= 2431:
            return True
    return False