import malaya

from malaya_tagging_pipeline import stem_tokens, tag_tokens, word_tokenize

# This text has been taken from the abstract of the Prostate Cancer Wikipedia page
# https://ms.wikipedia.org/wiki/Barah_prostat
test_text = """
Barah prostat atau kanser prostat ialah satu bentuk kanser yang berkembang di dalam prostat, satu kelenjar dalam sistem pembiakan jantan atau lelaki. Ia merupakan salah satu kanser yang paling biasa di kalangan lelaki, terutamanya yang berumur lebih 50 tahun.
"""

expected_output = [
"""
Barah   Barah   PROPN
prostat ostat   NOUN
atau    atau    CCONJ
kanser  kanser  NOUN
prostat ostat   NOUN
ialah   ialah   AUX
satu    satu    DET
bentuk  bentuk  NOUN
kanser  kanser  NOUN
yang    yang    PRON
berkembang      kembang VERB
di      k       ADP
dalam   dalam   ADP
prostat ostat   NOUN
,       ,       PUNCT
satu    satu    DET
kelenjar        kelenjar        NOUN
dalam   dalam   ADP
sistem  sistem  NOUN
pembiakan       biak    NOUN
jantan  jantan  NOUN
atau    atau    CCONJ
lelaki  lelaki  NOUN
.       .       PUNCT""",
"""
Ia      Ia      PRON
merupakan       rupa    VERB
salah   salah   DET
satu    satu    DET
kanser  kanser  NOUN
yang    yang    PRON
paling  paling  ADV
biasa   biasa   ADJ
di      k      ADP
kalangan        kalang  NOUN
lelaki  lelaki  NOUN
,       ,       PUNCT
terutamanya     utama   ADV
yang    yang    PRON
berumur umur    VERB
lebih   lebih   ADV
50      af      NUM
tahun   tahun   NOUN
.       .       PUNCT
"""
]

def expected_sentence_to_token_data(sentence: str) -> list[tuple[str, str, str]]:
    sentence_token_data = sentence.split("\n")
    expected_token_data: list[tuple[str, str, str]] = []
    for token_data in sentence_token_data:
        token_data = token_data.strip()
        if not token_data:
            continue
        token, lemma, pos_tag = token_data.split()
        expected_token_data.append((token, lemma, pos_tag))
    return expected_token_data

def test_system() -> None:
    sentence_splitter = malaya.tokenizer.SentenceTokenizer()
    word_tokenizer = malaya.tokenizer.Tokenizer()
    lemmatizer = malaya.stem.huggingface('mesolitica/stem-lstm-512', force_check=True)
    pos_tagger = malaya.pos.huggingface("mesolitica/pos-t5-small-standard-bahasa-cased", force_check=True)
    sentence_splits = sentence_splitter.tokenize(test_text)
    assert len(sentence_splits) == len(expected_output)
    for sentence_index, sentence in enumerate(sentence_splits):
        expected_sentence = expected_output[sentence_index]
        expected_token_data = expected_sentence_to_token_data(expected_sentence)

        tokens = word_tokenize(word_tokenizer, sentence, lowercase=False)
        lemmas  = stem_tokens(lemmatizer, tokens)
        pos_tags = tag_tokens(pos_tagger, tokens)

        assert len(tokens) == len(expected_token_data)
        assert len(lemmas) == len(expected_token_data)
        assert len(pos_tags) == len(expected_token_data)

        token_index = 0
        for token, lemma, pos_tag in zip(tokens, lemmas, pos_tags):
            print(token, lemma, pos_tag)
            assert token == expected_token_data[token_index][0]
            assert lemma == expected_token_data[token_index][1]
            assert pos_tag == expected_token_data[token_index][2]

            token_index += 1
            