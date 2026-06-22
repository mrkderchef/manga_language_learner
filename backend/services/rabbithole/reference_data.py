"""Static editorial language data used by the Rabbithole NLP pipeline.

These compact mappings are application fallbacks and learning annotations, not
authoritative replacements for Sudachi, JMdict, KANJIDIC2, or other cited data
sources. Keeping them here separates reference content from processing logic.
"""

from __future__ import annotations


FUNCTION_GLOSSARY = {
    "は": "topic marker",
    "が": "subject marker",
    "を": "object marker",
    "に": "to / at / in",
    "へ": "toward",
    "で": "location / means marker",
    "と": "and / with / quote",
    "も": "also",
    "の": "of / possessive",
    "から": "from / because",
    "まで": "until / up to",
    "より": "than / from",
    "や": "and / such as",
    "か": "question marker",
    "ね": "seeking agreement",
    "よ": "emphasis",
    "な": "soft emphasis",
    "ぞ": "emphatic assertion",
    "さ": "casual emphasis",
    "て": "and then / -te form",
    "た": "past",
    "だ": "copula",
    "です": "polite copula",
    "ます": "polite verb ending",
    "ない": "not",
    "ん": "explanatory / contraction",
    "のだ": "explanatory",
}

LEXICAL_GLOSSARY = {
    "夢": "dream",
    "果て": "the end",
    "さん": "honorific",
    "ちゃん": "affectionate suffix",
    "くん": "familiar honorific",
    "ガタッ": "clatter / thud",
    "ドキドキ": "heartbeat / nervous excitement",
}

POS_LABELS = {
    "名詞": "noun",
    "動詞": "verb",
    "形容詞": "adjective",
    "副詞": "adverb",
    "連体詞": "pre-noun adjective",
    "接続詞": "conjunction",
    "感動詞": "interjection",
    "助詞": "particle",
    "助動詞": "auxiliary",
    "接頭辞": "prefix",
    "接尾辞": "suffix",
    "補助記号": "symbol / punctuation",
}

GRAMMAR_DETAILS = {
    "は": {
        "label": "topic marker",
        "glosses": [
            "marks the sentence topic",
            "often adds contrast depending on context",
            "written は but pronounced わ in this particle use",
        ],
        "tags": ["grammar", "particle", "JLPT N5 core"],
    },
    "が": {
        "label": "subject marker",
        "glosses": [
            "marks grammatical subject",
            "often introduces new or focused information",
        ],
        "tags": ["grammar", "particle", "JLPT N5 core"],
    },
    "を": {
        "label": "object marker",
        "glosses": [
            "marks direct object",
            "pronounced お in particle use",
        ],
        "tags": ["grammar", "particle", "JLPT N5 core"],
    },
    "に": {
        "label": "target / time marker",
        "glosses": [
            "marks destination, indirect object, or point in time",
            "can also mark purpose with movement verbs",
        ],
        "tags": ["grammar", "particle"],
    },
    "で": {
        "label": "location / means marker",
        "glosses": [
            "marks location of action",
            "marks method, instrument, or material",
        ],
        "tags": ["grammar", "particle"],
    },
    "へ": {
        "label": "direction marker",
        "glosses": [
            "marks direction toward a destination",
            "written へ but pronounced え in this particle use",
        ],
        "tags": ["grammar", "particle"],
    },
    "の": {
        "label": "genitive linker",
        "glosses": [
            "links nouns (roughly \"of\")",
            "can nominalize clauses in some constructions",
        ],
        "tags": ["grammar", "particle", "JLPT N5 core"],
    },
    "と": {
        "label": "and / with / quotation marker",
        "glosses": [
            "joins nouns as \"and\"",
            "marks quoted speech or thought",
            "can mark companion as \"with\"",
        ],
        "tags": ["grammar", "particle"],
    },
    "も": {
        "label": "also marker",
        "glosses": [
            "adds meaning like \"also / too\"",
            "can replace は, が, or を depending on sentence role",
        ],
        "tags": ["grammar", "particle"],
    },
    "か": {
        "label": "question marker",
        "glosses": [
            "marks a direct question",
            "can also create alternatives or indefinites",
        ],
        "tags": ["grammar", "particle"],
    },
    "から": {
        "label": "from / because marker",
        "glosses": [
            "marks origin or starting point",
            "can mark reason (\"because\")",
        ],
        "tags": ["grammar", "particle"],
    },
    "まで": {
        "label": "until / up to marker",
        "glosses": [
            "marks end point in time or space",
            "often pairs with から",
        ],
        "tags": ["grammar", "particle"],
    },
    "より": {
        "label": "comparison marker",
        "glosses": [
            "marks comparison baseline (\"than\")",
            "can mark source in formal/literary style",
        ],
        "tags": ["grammar", "particle"],
    },
    "や": {
        "label": "non-exhaustive listing marker",
        "glosses": [
            "lists representative items (\"A, B, and so on\")",
        ],
        "tags": ["grammar", "particle"],
    },
    "ね": {
        "label": "agreement-seeking ending",
        "glosses": [
            "softly seeks listener agreement",
        ],
        "tags": ["grammar", "sentence ending"],
    },
    "よ": {
        "label": "assertive ending",
        "glosses": [
            "adds emphasis or new information for the listener",
        ],
        "tags": ["grammar", "sentence ending"],
    },
    "だ": {
        "label": "plain copula",
        "glosses": [
            "plain assertive copula",
            "used in casual/plain style",
        ],
        "tags": ["grammar", "auxiliary", "copula"],
    },
    "です": {
        "label": "polite copula",
        "glosses": [
            "polite copula",
            "used to raise formality",
        ],
        "tags": ["grammar", "auxiliary", "copula"],
    },
    "ます": {
        "label": "polite verb ending",
        "glosses": [
            "politeness marker attached to verb stem",
            "appears in non-past affirmative polite forms",
        ],
        "tags": ["grammar", "auxiliary", "politeness"],
    },
    "ない": {
        "label": "negative auxiliary",
        "glosses": [
            "marks negation for verbs/adjectival predicates",
        ],
        "tags": ["grammar", "auxiliary", "negation"],
    },
    "た": {
        "label": "past/perfect auxiliary",
        "glosses": [
            "marks past or completed action/state",
        ],
        "tags": ["grammar", "auxiliary"],
    },
    "て": {
        "label": "te-form linker",
        "glosses": [
            "links actions/clauses",
            "can support requests, progressive forms, and many fixed constructions",
        ],
        "tags": ["grammar", "verb form"],
    },
}

SYMBOL_DETAILS = {
    "、": {
        "label": "Japanese comma",
        "glosses": ["pause separator in Japanese writing"],
        "tags": ["symbol", "punctuation"],
    },
    "。": {
        "label": "Japanese period",
        "glosses": ["sentence terminator in Japanese writing"],
        "tags": ["symbol", "punctuation"],
    },
    "「": {
        "label": "opening quote",
        "glosses": ["opens Japanese quotation marks"],
        "tags": ["symbol", "punctuation"],
    },
    "」": {
        "label": "closing quote",
        "glosses": ["closes Japanese quotation marks"],
        "tags": ["symbol", "punctuation"],
    },
    "『": {
        "label": "opening inner quote",
        "glosses": ["opens nested/emphatic Japanese quotes"],
        "tags": ["symbol", "punctuation"],
    },
    "』": {
        "label": "closing inner quote",
        "glosses": ["closes nested/emphatic Japanese quotes"],
        "tags": ["symbol", "punctuation"],
    },
    "・": {
        "label": "middle dot",
        "glosses": ["separator in names or borrowed compounds"],
        "tags": ["symbol", "punctuation"],
    },
    "ー": {
        "label": "long vowel mark",
        "glosses": ["extends the previous vowel sound"],
        "tags": ["symbol", "kana mark"],
    },
    "…": {
        "label": "ellipsis",
        "glosses": ["pause, trailing thought, or silence"],
        "tags": ["symbol", "punctuation"],
    },
    "？": {
        "label": "question mark",
        "glosses": ["question punctuation"],
        "tags": ["symbol", "punctuation"],
    },
    "！": {
        "label": "exclamation mark",
        "glosses": ["exclamatory punctuation"],
        "tags": ["symbol", "punctuation"],
    },
}

SMALL_KANA_NOTES = {
    "ぁ": "small a kana; used in stylistic spellings/phonetic effects",
    "ぃ": "small i kana; used in stylistic spellings/phonetic effects",
    "ぅ": "small u kana; used in stylistic spellings/phonetic effects",
    "ぇ": "small e kana; used in stylistic spellings/phonetic effects",
    "ぉ": "small o kana; used in stylistic spellings/phonetic effects",
    "ゃ": "small ya kana; combines with i-row kana for contracted sounds",
    "ゅ": "small yu kana; combines with i-row kana for contracted sounds",
    "ょ": "small yo kana; combines with i-row kana for contracted sounds",
    "ゎ": "small wa kana; rare in modern standard text",
    "っ": "small tsu (sokuon); marks consonant gemination",
}

KANA_ROMAJI = {
    **dict(zip("あいうえお", ("a", "i", "u", "e", "o"))),
    **dict(zip("かきくけこ", ("ka", "ki", "ku", "ke", "ko"))),
    **dict(zip("がぎぐげご", ("ga", "gi", "gu", "ge", "go"))),
    **dict(zip("さしすせそ", ("sa", "shi", "su", "se", "so"))),
    **dict(zip("ざじずぜぞ", ("za", "ji", "zu", "ze", "zo"))),
    **dict(zip("たちつてと", ("ta", "chi", "tsu", "te", "to"))),
    **dict(zip("だぢづでど", ("da", "ji", "zu", "de", "do"))),
    **dict(zip("なにぬねの", ("na", "ni", "nu", "ne", "no"))),
    **dict(zip("はひふへほ", ("ha", "hi", "fu", "he", "ho"))),
    **dict(zip("ばびぶべぼ", ("ba", "bi", "bu", "be", "bo"))),
    **dict(zip("ぱぴぷぺぽ", ("pa", "pi", "pu", "pe", "po"))),
    **dict(zip("まみむめも", ("ma", "mi", "mu", "me", "mo"))),
    **dict(zip("やゆよ", ("ya", "yu", "yo"))),
    **dict(zip("らりるれろ", ("ra", "ri", "ru", "re", "ro"))),
    "わ": "wa", "ゐ": "i", "ゑ": "e", "を": "o", "ん": "n",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o", "ゎ": "wa",
    "ゔ": "vu",
}

KANA_DIGRAPH_ROMAJI = {
    **{f"{lead}{small}": roman for lead, values in {
        "き": ("kya", "kyu", "kyo"), "ぎ": ("gya", "gyu", "gyo"),
        "し": ("sha", "shu", "sho"), "じ": ("ja", "ju", "jo"),
        "ち": ("cha", "chu", "cho"), "に": ("nya", "nyu", "nyo"),
        "ひ": ("hya", "hyu", "hyo"), "び": ("bya", "byu", "byo"),
        "ぴ": ("pya", "pyu", "pyo"), "み": ("mya", "myu", "myo"),
        "り": ("rya", "ryu", "ryo"),
    }.items() for small, roman in zip("ゃゅょ", values)},
    "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo",
    "しぇ": "she", "じぇ": "je", "ちぇ": "che",
}


__all__ = [
    "FUNCTION_GLOSSARY",
    "GRAMMAR_DETAILS",
    "KANA_DIGRAPH_ROMAJI",
    "KANA_ROMAJI",
    "LEXICAL_GLOSSARY",
    "POS_LABELS",
    "SMALL_KANA_NOTES",
    "SYMBOL_DETAILS",
]
