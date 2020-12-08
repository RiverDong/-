import os
import re
import pandas as pd
import numpy as np
from string import Template
from collections import OrderedDict
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet



# Noise ######################################################
# In DATE regex, mm and dd can have only one digit, and / can also be - or \s
all_month_digit = r'(1[012]|0?[1-9])'
nofeb_month_digit = r'(1[012]|0?[13456789])'
long_month_digit = r'(1[02]|0?[13578])'
all_month_letter = r'(January|Jan\.?|February|Feb\.?|March|Mar\.?|April|Apr\.?|May|June|Jun\.?|July|Jul\.?|August|Aug\.?|September|Sept\.?|Sep\.?|October|Oct\.?|November|Nov\.?|December|Dec\.?)'
nofeb_month_letter = r'(January|Jan\.?|March|Mar\.?|April|Apr\.?|May|June|Jun\.?|July|Jul\.?|August|Aug\.?|September|Sept\.?|Sep\.?|October|Oct\.?|November|Nov\.?|December|Dec\.?)'
long_month_letter = r'(January|Jan\.?|March|Mar\.?|May|July|Jul\.?|August|Aug\.?|October|Oct\.?|December|Dec\.?)'

mm_dd_yyyy = Template(r'(?ias)${prefix}((${all_month}${sep1}(1\d|2[0-8]|0?[1-9])${day_suffix}|${nofeb_month}${sep1}'
                      r'(29|30)${day_suffix}|${long_month}${sep1}31${day_suffix})${sep2}(19|20|21)\d{2}|0?2${sep1}'
                      r'29${day_suffix}${sep2}((19|20|21)(0[48]|[2468][048]|[13579][26])|2000))${suffix}')
yyyy_mm_dd = Template(r'(?ias)${prefix}((19|20|21)\d{2}${sep1}(${all_month}${sep2}(1\d|2[0-8]|0?[1-9])${day_suffix}|'
                      r'${nofeb_month}${sep2}(29|30)${day_suffix}|${long_month}${sep2}31${day_suffix})|((19|20|21)(0[48]|'
                      r'[2468][048]|[13579][26])|2000)${sep1}0?2${sep2}29${day_suffix})${suffix}')
mm_yyyy = Template(r'(?ias)${prefix}(${all_month}${sep1}(19|20|21)\d{2})${suffix}')
yyyy_mm = Template(r'(?ias)${prefix}((19|20|21)\d{2}${sep1}${all_month})${suffix}')
mm_dd_yy = Template(r'(?ias)${prefix}((${all_month}${sep1}(1\d|2[0-8]|0?[1-9])${day_suffix}|${nofeb_month}${sep1}'
                    r'(29|30)${day_suffix}|${long_month}${sep1}31${day_suffix})${sep2}\d{2}|0?2${sep1}29${day_suffix}'
                    r'${sep2}((0[48]|[2468][048]|[13579][26])|00))${suffix}')
yy_mm_dd = Template(r'(?ias)${prefix}(\d{2}${sep1}(${all_month}${sep2}(1\d|2[0-8]|0?[1-9])${day_suffix}|${nofeb_month}'
                    r'${sep2}(29|30)${day_suffix}|${long_month}${sep2}31${day_suffix})|((0[48]|[2468][048]|[13579][26])|00)'
                    r'${sep1}0?2${sep2}29${day_suffix})${suffix}')
mm_dd = Template(r'(?ias)${prefix}(${all_month}${sep1}(1\d|2\d|0?[1-9])${day_suffix}|${nofeb_month}${sep1}30${day_suffix}|'
                 r'${long_month}${sep1}31${day_suffix})${suffix}')
dd_mm = Template(r'(?ias)${prefix}((1\d|2\d|0?[1-9])${day_suffix}${sep1}${all_month}|30${day_suffix}${sep1}${nofeb_month}|'
                 r'31${day_suffix}${sep1}${long_month})${suffix}')
mm_yy = Template(r'(?ias)${prefix}(${all_month}${sep1}\d{2})${suffix}')
yy_mm = Template(r'(?ias)${prefix}(\d{2}${sep1}${all_month})${suffix}')

noise_marker = 'noisemark'
noise_regex = OrderedDict([
    ('URL', (re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'), )),
    ('EMAILADDRESS', (re.compile(r'(?ias)([a-zA-Z\d_+.-]+@[a-zA-Z\d-]+\.[a-zA-Z\d.-]+)'), )),  # https://emailregex.com/
    ('EMOJI', (re.compile(r'(?ias)(?<![:()-])(:\s*-?\s*[()])(?![:()-])'), )),
    ('TIME', (re.compile(r'(?ias)(?<![\d$=<>:*/+-])((1[0-2]|0?[1-9])(\s*:\s*[0-5]\d){0,2}\s*(a\.?m\.?|p\.?m\.?)|(1\d|2[0-3]|0?\d)'
                         r'(\s*:\s*[0-5]\d){1,2})(?![\d%=<>:*/+-])'), )),
    ('ORDERNUMBER', (re.compile(r'(?ias)(?<![\d$=<>*/+-])(\d{3}-\d{7}-\d{7})(?![\d%=<>*/+-])'), )),
    ('PHONENUMBER', (re.compile(r'(?ias)(?<![\d$=<>()*/+-])((\+?\d(\s|-)?)?(\(\d{3}\)|\d{3})(\s|-)?\d{3}(\s|-)?\d{4})(?![\d%=<>()*/+-])'), )),
    ('DATE', (re.compile(mm_dd_yyyy.substitute(all_month=all_month_digit, nofeb_month=nofeb_month_digit, long_month=long_month_digit,
                                               sep1=r'([\s/-])', sep2=r'([\s/-])', prefix=r'(?<![\d$=<>*/+-])', suffix=r'(?![\d%=<>*/+-])', day_suffix=r'')),  # mm/dd/yyyy from 1900 to 2199
              re.compile(yyyy_mm_dd.substitute(all_month=all_month_digit, nofeb_month=nofeb_month_digit, long_month=long_month_digit,
                                               sep1=r'([\s/-])', sep2=r'([\s/-])', prefix=r'(?<![\d$=<>*/+-])', suffix=r'(?![\d%=<>*/+-])', day_suffix=r'')),  # yyyy/mm/dd from 1900 to 2199
              re.compile(mm_yyyy.substitute(all_month=all_month_digit, sep1=r'([\s/-])', prefix=r'(?<![\d$=<>*/+-])', suffix=r'(?![\d%=<>*/+-])')),  # mm/yyyy from 1900 to 2199
              re.compile(yyyy_mm.substitute(all_month=all_month_digit, sep1=r'([\s/-])', prefix=r'(?<![\d$=<>*/+-])', suffix=r'(?![\d%=<>*/+-])')),  # yyyy/mm from 1900 to 2199
              re.compile(mm_dd_yy.substitute(all_month=all_month_digit, nofeb_month=nofeb_month_digit, long_month=long_month_digit,
                                             sep1=r'([\s/-])', sep2=r'([\s/-])', prefix=r'(?<![\d$=<>*/+-])', suffix=r'(?![\d%=<>*/+-])', day_suffix=r'')),  # mm/dd/yy from 2000 to 2099
              re.compile(yy_mm_dd.substitute(all_month=all_month_digit, nofeb_month=nofeb_month_digit, long_month=long_month_digit,
                                             sep1=r'([\s/-])', sep2=r'([\s/-])', prefix=r'(?<![\d$=<>*/+-])', suffix=r'(?![\d%=<>*/+-])', day_suffix=r'')),  # yy/mm/dd from 2000 to 2099
              re.compile(mm_dd.substitute(all_month=all_month_digit, nofeb_month=nofeb_month_digit, long_month=long_month_digit,
                                          sep1=r'([\s/-])', prefix=r'(?<![\d$=<>*/+-])', suffix=r'(?![\d%=<>*/+-])', day_suffix=r'')),  # mm/dd
              re.compile(mm_yy.substitute(all_month=all_month_digit, sep1=r'([\s/-])', prefix=r'(?<![\d$=<>*/+-])', suffix=r'(?![\d%=<>*/+-])')),  # mm/yy from 2000 to 2099
              re.compile(yy_mm.substitute(all_month=all_month_digit, sep1=r'([\s/-])', prefix=r'(?<![\d$=<>*/+-])', suffix=r'(?![\d%=<>*/+-])')),  # yy/mm from 2000 to 2099
              re.compile(mm_dd_yyyy.substitute(all_month=all_month_letter, nofeb_month=nofeb_month_letter, long_month=long_month_letter,
                                               sep1=r'(\s*-\s*|\s*)', sep2=r'(\s*,\s*|\s+)', prefix=r'(?<![a-zA-Z])', suffix=r'(?![\d%=<>*/+-])', day_suffix=r'\s*(st|nd|rd|th)?')),  # MON dd, yyyy from 1900 to 2199
              re.compile(mm_yyyy.substitute(all_month=all_month_letter, sep1=r'(\s*,\s*|\s*)', prefix=r'(?<![a-zA-Z])', suffix=r'(?![\d%=<>*/+-])')),  # MON, yyyy from 1900 to 2199
              re.compile(mm_dd.substitute(all_month=all_month_letter, nofeb_month=nofeb_month_letter, long_month=long_month_letter,
                                          sep1=r'(\s*-\s*|\s*)', prefix=r'(?<![a-zA-Z])', suffix=r'(?![a-zA-Z\d%=<>*/+-])', day_suffix=r'\s*(st|nd|rd|th)?')),  # MON dd
              re.compile(dd_mm.substitute(all_month=all_month_letter, nofeb_month=nofeb_month_letter, long_month=long_month_letter,
                                          sep1=r'(\s*-\s*|\s*)', prefix=r'(?<![\d$=<>*/+-])', suffix=r'(?![a-zA-Z])', day_suffix=r'\s*(st|nd|rd|th)?'))  # dd MON
              )),
    ('ORDINALNUMBER', (re.compile(r'(?ias)(\d+\s*(st|nd|rd|th)\.)'), re.compile(r'(?ias)(\d+\s*(st|nd|rd|th))(?![a-zA-Z])')))
])

# Scrub token ######################################################
scrub_token_marker = noise_marker
scrub_token_collapse = True
scrub_token_regex = re.compile(r'(?as)(<(ACCOUNT_NUMBER|APPLICATION_DEVICE_USAGE|AWARDS_PROGRAM|CREDENTIALS_DECRYPTION_KEYS|'
                               r'CREDIT_CARD_NUMBER|CREDIT_CARD_CVC|CUSTOMER_CREDENTIALS|DATE_OF_BIRTH|EMAIL_ADDRESS|'
                               r'GIFT_MESSAGE|HOME_SERVICES|INTERNATIONAL_DRIVERS_LICENSE|MAILING_ADDRESS|NAME|'
                               r'NATIONAL_IDENTIFIER|NICK_NAME|OTHER_GOVERNMENT_IDENTIFICATION|PASSPORT|PHONE_NUMBER|'
                               r'URL|USERNAME|POTENTIAL_UCI)>|\u2022)')

# Acronym ####################################################
# Acronym is often separated by periods. We keep a list of common acronym to
# (1) regard "t.v." and "tv" as the same thing
# (2) avoid separating "t.v." as "t v " after removing the punctuation
# (3) avoid changing e.g. "3D" to "numbermarker D"
# To match the longest acronym if possible, we sort the acronyms descendingly in length
# To avoid matching non-acronym words, we
# (1) exclude the acronym of length 1
# (2) match the acronym only when it is not adjacent to any letters and digits
acronym_marker = ''
# common_acronym = pd.read_csv(os.path.join(input_path, 'common_acronym.txt'),
#                              sep='\t', index_col=False, header=None, dtype=str).iloc[:, 0]
# common_acronym_sorted = sorted(common_acronym.tolist(), key=len, reverse=True)
# acronym_regex = re.compile(r'(?ias)(?<![\da-zA-Z])(' +
#                            r'|'.join([r'\.? *'.join(list(i)) + r'\.? *' for i in common_acronym_sorted if len(i) > 1]) +
#                            r'|(?-i:U\.? *S\.? *|U\.? *N\.? *)' +
#                            r')(?![\da-zA-Z])')
acronym_regex = ''

# Number #####################################################
number_marker = 'numbermark'
number_regex = re.compile(r'(?ias)([+-]?\s*(\d*\.\d+|\d+)\s*(e\s*[+-]?\s*\d+)?)')

# Apostrophe contract ########################################
apostr_contract_collapse = False  # Important: always let this be "False" otherwise, e.g., stemmer_or_lemmatizer('wewill') -> 'wewil'
apostr_contract_regex = OrderedDict([
    ("dont", (re.compile(r"(?ias)(?<![a-zA-Z])(dont)(?![a-zA-Z])"), "do", "not", 2)),
    ("didnt", (re.compile(r"(?ias)(?<![a-zA-Z])(didnt)(?![a-zA-Z])"), "did", "not", 2)),
    ("cann't", (re.compile(r"(?ias)(?<![a-zA-Z'\u2019])(cann['\u2019]t)(?![a-zA-Z'\u2019])"), "can", "not", 3)),
    ("can't", (re.compile(r"(?ias)(?<![a-zA-Z'\u2019])(can['\u2019]t)(?![a-zA-Z'\u2019])"), "can", "not", 3)),
    ("shan't", (re.compile(r"(?ias)(?<![a-zA-Z'\u2019])(shan['\u2019]t)(?![a-zA-Z'\u2019])"), "shall", "not", 3)),
    ("won't", (re.compile(r"(?ias)(?<![a-zA-Z'\u2019])(won['\u2019]t)(?![a-zA-Z'\u2019])"), "will", "not", 3)),
    ("n't", (re.compile(r"(?ias)(?<![a-zA-Z'\u2019])([a-zA-Z]+n['\u2019]t)(?![a-zA-Z'\u2019])"), None, "not", 3)),
    ("'d", (re.compile(r"(?ias)(?<![a-zA-Z'\u2019])([a-zA-Z]+['\u2019]d)(?![a-zA-Z'\u2019])"), None, "would", 2)),
    ("'ve", (re.compile(r"(?ias)(?<![a-zA-Z'\u2019])([a-zA-Z]+['\u2019]ve)(?![a-zA-Z'\u2019])"), None, "have", 3)),
    ("'re", (re.compile(r"(?ias)(?<![a-zA-Z'\u2019])([a-zA-Z]+['\u2019]re)(?![a-zA-Z'\u2019])"), None, "are", 3)),
    ("'ll", (re.compile(r"(?ias)(?<![a-zA-Z'\u2019])([a-zA-Z]+['\u2019]ll)(?![a-zA-Z'\u2019])"), None, "will", 3)),
    ("'m", (re.compile(r"(?ias)(?<![a-zA-Z'\u2019])([a-zA-Z]+['\u2019]m)(?![a-zA-Z'\u2019])"), None, "am", 2)),
    ("'s", (re.compile(r"(?ias)(?<![a-zA-Z'\u2019])([a-zA-Z]+['\u2019]s)(?![a-zA-Z'\u2019])"), None, "is", 2))
])

# Apostrophe possess #########################################
apostr_possess_marker = ''
apostr_possess_collapse = False
apostr_possess_regex = re.compile(r"(?ias)(?<![a-zA-Z'\u2019])([a-zA-Z]+s['\u2019])(?![a-zA-Z'\u2019])")

# Hyphen #####################################################
hyphen_collapse = True
hyphen_regex = re.compile(r'(?ias)(?<![a-zA-Z-])([a-zA-Z]+-[a-zA-Z]+)(?![a-zA-Z-])')

# Punctuation ################################################
punc_trans = dict()

# Final replacement ##########################################
final_replacement = {'please': re.compile(r'(?ias)(?<![a-zA-Z])(pls)(?![a-zA-Z])'),
                     'just': re.compile(r'(?ias)(?<![a-zA-Z])(jst)(?![a-zA-Z])'),
                     'thanks': re.compile(r'(?ias)(?<![a-zA-Z])(thx)(?![a-zA-Z])')}

# Spelling correction ########################################
# max_dictionary_edit_distance: Maximum edit distance for doing lookups
# prefix_length: The length of word prefixes used for spell checking
sym_spell = None
max_lookup_edit_distance = None

# Word removal ###############################################
remove_created_words = False

unit_set = {'mm', 'cm', 'dm', 'm', 'km', 'ft', 'yd', 'mi', 'sqm', 'sq', 'ml', 'cl', 'dl', 'l', 'fl', 'oz', 'qt', 'gal',
            's', 'h', 'd', 'mo', 'yr', 'yrs', 'hz', 'db', 'mph', 'kph', 'rpm', 'mg', 'g', 'kg', 't', 'lb', 'n',
            'j', 'w', 'kw', 'kwh', 'pa', 'psi', 'k', 'cal', 'kcal', 'c', 'f', 'lm', 'lx', 'ls', 'v', 'mol'}
stopword_set = set(stopwords.words('english'))

word_to_remove = {'www', 'com', 'urlnoisemark', 'emailaddressnoisemark', 'emojinoisemark', 'timenoisemark',
                  'ordernumbernoisemark', 'phonenumbernoisemark', 'datenoisemark', 'ordinalnumbernoisemark',
                  'numbermark'} | unit_set
word_to_remove_bm25 = {'amazon'} | stopword_set

# Stemming or lemmatization ##################################
stemmer_or_lemmatizer = SnowballStemmer('english', ignore_stopwords=False).stem
# stemmer_or_lemmatizer = lambda x: WordNetLemmatizer().lemmatize(x, pos='v')

