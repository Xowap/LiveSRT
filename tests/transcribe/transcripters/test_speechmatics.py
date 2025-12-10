from typing import Literal

import pytest

from livesrt.transcribe.transcripters.speechmatics import (
    LRI,
    PDI,
    RLI,
    TempWord,
    TranscriptBuilder,
    WordAlternative,
    WordDisplay,
    join_words,
)


def tokenize(
    text: list[str],
    lang: str,
    speaker: str = "A",
    direction: Literal["ltr", "rtl"] = "ltr",
) -> list[TempWord]:
    out = []

    for i, w in enumerate(text):
        attachment = {
            (False, False): "both",
            (False, True): "previous",
            (True, False): "next",
            (True, True): "none",
        }[
            (
                w[0].isspace(),
                w[-1].isspace(),
            )
        ]

        out.append(
            TempWord(
                type=("word" if w.strip()[0].isalnum() else "punctuation"),
                start_time=float(i),
                end_time=float(i + 1),
                attaches_to=attachment,
                is_eos=w.strip() in ".?!",
                entity_class="",
                alternatives=[
                    WordAlternative(
                        content=w.strip(),
                        confidence=1.0,
                        language=lang,
                        display=WordDisplay(direction),
                        speaker=speaker,
                        tags=[],
                    )
                ],
            )
        )

    return out


@pytest.fixture
def tb_1():
    return tokenize([" Vive ", " les ", " frites ", " ! "], "fr")


@pytest.fixture
def tb_1b():
    return tokenize([" Veev ", " ley ", " freet ", " ! "], "fr")


@pytest.fixture
def tb_2():
    return tokenize([" I ", " love ", " potatoes", "! "], "en")


def test_tokenize(tb_1: list[TempWord], tb_2: list[TempWord]):
    assert [x.alternatives[0].content for x in tb_1] == ["Vive", "les", "frites", "!"]
    assert [x.attaches_to for x in tb_1] == ["none", "none", "none", "none"]

    assert [x.alternatives[0].content for x in tb_2] == ["I", "love", "potatoes", "!"]
    assert [x.attaches_to for x in tb_2] == ["none", "none", "next", "previous"]


@pytest.fixture
def tb_rtl():
    """Pure RTL text (e.g., Hebrew/Arabic)"""
    return tokenize([" שלום ", " עולם ", " ! "], "he", direction="rtl")


@pytest.fixture
def tb_bidi_ltr_first():
    """BiDi: LTR then RTL (English then Hebrew)"""
    words = []
    # English words (LTR)
    words.extend(tokenize([" Hello ", " world "], "en", direction="ltr"))
    # Hebrew words (RTL)
    words.extend(tokenize([" שלום ", " עולם "], "he", direction="rtl"))
    return words


@pytest.fixture
def tb_bidi_rtl_first():
    """BiDi: RTL then LTR (Hebrew then English)"""
    words = []
    # Hebrew words (RTL)
    words.extend(tokenize([" שלום ", " עולם "], "he", direction="rtl"))
    # English words (LTR)
    words.extend(tokenize([" Hello ", " world "], "en", direction="ltr"))
    return words


@pytest.fixture
def tb_bidi_mixed():
    """BiDi: Multiple direction changes"""
    words = []
    words.extend(tokenize([" Hello "], "en", direction="ltr"))
    words.extend(tokenize([" שלום "], "he", direction="rtl"))
    words.extend(tokenize([" world "], "en", direction="ltr"))
    words.extend(tokenize([" עולם "], "he", direction="rtl"))
    return words


@pytest.fixture
def tb_rtl_with_punctuation():
    """RTL text with punctuation that attaches"""
    return tokenize([" مرحبا ", " بك", "! "], "ar", direction="rtl")


def test_join_words(tb_1, tb_2):
    assert join_words(tb_1) == "Vive les frites !"
    assert join_words(tb_2) == "I love potatoes!"


def test_join_words_rtl(tb_rtl):
    """Test pure RTL text with default natural_direction='ltr'"""
    result = join_words(tb_rtl)
    # Since natural_direction is 'ltr' and text is 'rtl', it should be wrapped
    assert result == f"{RLI}שלום עולם !{PDI}"


def test_join_words_rtl_natural(tb_rtl):
    """Test pure RTL text with natural_direction='rtl'"""
    result = join_words(tb_rtl, natural_direction="rtl")
    # Since natural_direction is 'rtl' and text is 'rtl', no wrapping needed
    assert result == "שלום עולם !"


def test_join_words_bidi_ltr_first(tb_bidi_ltr_first):
    """Test BiDi text starting with LTR"""
    result = join_words(tb_bidi_ltr_first)
    # LTR segment first (no wrap), then RTL segment (wrapped)
    assert result == f"Hello world {RLI}שלום עולם{PDI}"


def test_join_words_bidi_rtl_first(tb_bidi_rtl_first):
    """Test BiDi text starting with RTL"""
    result = join_words(tb_bidi_rtl_first)
    # RTL segment first (wrapped), then LTR segment (wrapped)
    assert result == f"{RLI}שלום עולם{PDI} {LRI}Hello world{PDI}"


def test_join_words_bidi_rtl_first_natural_rtl(tb_bidi_rtl_first):
    """Test BiDi text starting with RTL, with natural_direction='rtl'"""
    result = join_words(tb_bidi_rtl_first, natural_direction="rtl")
    # RTL segment first (no wrap), then LTR segment (wrapped)
    assert result == f"שלום עולם {LRI}Hello world{PDI}"


def test_join_words_bidi_mixed(tb_bidi_mixed):
    """Test BiDi text with multiple direction changes"""
    result = join_words(tb_bidi_mixed)
    # LTR first (no wrap), RTL (wrapped), LTR (wrapped), RTL (wrapped)
    assert result == f"Hello {RLI}שלום{PDI} {LRI}world{PDI} {RLI}עולם{PDI}"


def test_join_words_rtl_with_punctuation(tb_rtl_with_punctuation):
    """Test RTL text with attaching punctuation"""
    result = join_words(tb_rtl_with_punctuation)
    # Punctuation should attach without space
    assert result == f"{RLI}مرحبا بك!{PDI}"


def test_join_words_empty():
    """Test empty word list"""
    assert join_words([]) == ""


def test_join_words_single_word():
    """Test single word"""
    words = tokenize([" Hello "], "en", direction="ltr")
    assert join_words(words) == "Hello"


def test_join_words_single_rtl_word():
    """Test single RTL word"""
    words = tokenize([" שלום "], "he", direction="rtl")
    assert join_words(words) == f"{RLI}שלום{PDI}"

    # With natural RTL direction
    assert join_words(words, natural_direction="rtl") == "שלום"


def test_transcript_builder(tb_1: list[TempWord], tb_1b: list[TempWord]):
    tb = TranscriptBuilder(tb_1b, tb_1[:1])
    assert tb.generate() == "Vive ley freet !"

    tb.add_words("total", tb_1[:2])
    assert tb.generate() == "Vive les freet !"
