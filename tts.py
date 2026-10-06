"""TTS lingala : Simba-TTS-lin (VITS mono-locuteur, 16 kHz).

Le vocabulaire du modèle fait 39 signes : a-z SANS le x, les chiffres, l'espace,
l'apostrophe et le tiret. Donc ni accents, ni ɛ/ɔ, ni ponctuation. Or le lingala
s'écrit avec ɛ, ɔ et des tons (bísó, kokóma) et c'est exactement ce que GPT-4o
produit. C'est là la vraie cause des plantages : le tokenizer jette tout ce qui
sort du vocabulaire, les mots sortent estropiés et une phrase qui n'était que
ponctuation devient vide — d'où le RuntimeError dans narrow().

On normalise donc AVANT, puis on garde quand même le garde-fou de longueur et le
try/except : trois filets, pas un à la place des autres.
"""

import asyncio
import logging
import re
import unicodedata

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

from config import cpu_slot, settings

log = logging.getLogger(__name__)

# En dessous, VITS plante dans le calcul d'attention relative (narrow(): length
# must be non-negative).
MIN_LEN = 4
SILENCE_MS = 300  # respiration entre deux phrases, ça sonne beaucoup moins robot

# Substitutions AVANT filtrage : sans elles, « bisó bazalí ɛ́ » perd la moitié de
# ses lettres au lieu de devenir « biso bazali e ».
SUBS = {
    "ɛ": "e", "ɔ": "o", "ŋ": "ng", "ɲ": "ny", "x": "ks",
    "ʼ": "'", "’": "'", "‘": "'", "`": "'",
    "…": " ", "–": " ", "—": " ", "«": " ", "»": " ", "“": " ", "”": " ",
}

# Les chiffres SONT dans le vocabulaire, mais « 2 » y est aussi le pad_token
# (voir tokenizer_config.json) : il serait synthétisé comme du silence. On les
# épelle en lingala. Le prompt système demande déjà au LLM d'écrire les nombres
# en mots ; ceci rattrape les numéros de téléphone et les dates qui passent.
DIGITS = {
    "0": " zero ", "1": " moko ", "2": " mibale ", "3": " misato ", "4": " minei ",
    "5": " mitano ", "6": " motoba ", "7": " nsambo ", "8": " mwambe ", "9": " libwa ",
}

# Miroir de vocab.json (noter l'absence du x, traité par SUBS). Vérifié contre le
# tokenizer réel au chargement.
ALLOWED = set("abcdefghijklmnopqrstuvwyz '-")

_model: VitsModel | None = None
_tokenizer = None
_silence: np.ndarray | None = None


def normalize(text: str) -> str:
    """Ramène le texte à ce que le modèle sait prononcer."""
    text = text.lower()
    for src, dst in SUBS.items():
        text = text.replace(src, dst)
    # NFD + suppression des marques combinantes : retire les tons (á, à, ǎ) sans
    # toucher à la lettre porteuse.
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )
    text = "".join(DIGITS.get(c, c) for c in text)
    text = "".join(c if c in ALLOWED else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


def load() -> None:
    global _model, _tokenizer, _silence
    _tokenizer = AutoTokenizer.from_pretrained(settings.tts_model)
    _model = VitsModel.from_pretrained(settings.tts_model).eval()
    _silence = np.zeros(settings.sample_rate * SILENCE_MS // 1000, dtype=np.float32)

    if missing := ALLOWED - set(_tokenizer.get_vocab()):
        log.warning(
            "ALLOWED contient des caractères absents du vocabulaire de %s : %r "
            "— ils seront prononcés en silence.",
            settings.tts_model, sorted(missing),
        )
    log.info(
        "TTS chargé : %s (%d Hz, %d locuteur)",
        settings.tts_model, _model.config.sampling_rate, _model.config.num_speakers,
    )


def _generate(sentence: str) -> np.ndarray | None:
    """Une phrase -> PCM int16, ou None si elle n'est pas synthétisable."""
    text = normalize(sentence)
    if len(text) < MIN_LEN:
        log.debug("phrase ignorée (%d car. après normalisation) : %r", len(text), sentence)
        return None
    try:
        inputs = _tokenizer(text, return_tensors="pt")
        if inputs["input_ids"].shape[1] == 0:
            log.debug("tokenizer vide sur %r", text)
            return None
        with torch.no_grad():
            waveform = _model(**inputs).waveform
    except RuntimeError as exc:
        # Une phrase qui coince ne doit pas emporter le reste de la réponse.
        log.warning("VITS a échoué sur %r : %s", text, exc)
        return None

    audio = waveform.squeeze().numpy().astype(np.float32)
    if (peak := np.abs(audio).max()) > 0:
        # ponytail: normalisation par phrase, le volume peut légèrement respirer
        # d'une phrase à l'autre. Si ça s'entend, passer à un gain fixe mesuré.
        audio = audio / peak * 0.95
    audio = np.concatenate([audio, _silence])
    return (audio * 32767).astype(np.int16)


async def synthesize(sentence: str) -> np.ndarray | None:
    """Comme _generate, mais sans bloquer l'event loop ni saturer les vCPU."""
    async with cpu_slot:
        return await asyncio.to_thread(_generate, sentence)


if __name__ == "__main__":
    # Self-check de la normalisation : c'est elle qui décide si VITS plante.
    assert normalize("Bisó tozalí malámu.") == "biso tozali malamu"
    assert normalize("Nyɔ́nsɔ ezalí ɛ́lɔkɔ") == "nyonso ezali eloko"
    assert normalize('"Mbote!"') == "mbote"
    assert normalize("Tango ya 25") == "tango ya mibale mitano"
    assert normalize("...") == ""  # ne doit PAS atteindre le modèle
    assert normalize("  ") == ""
    assert normalize("Taxi") == "taksi"
    assert len(normalize("Té.")) < MIN_LEN  # trop court -> écarté, pas de crash
    assert ALLOWED.isdisjoint("x0123456789.!?"), "le filtre laisserait passer du hors-vocab"
    print("tts.py: normalisation OK")
