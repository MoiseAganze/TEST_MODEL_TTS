"""Configuration : tout vient de l'environnement, rien n'est codé en dur."""

import asyncio

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (OpenRouter) ---
    openrouter_api_key: str = ""
    llm_model: str = "openai/gpt-4o"

    # --- Modèles locaux ---
    # Simba-H (AfriHuBERT, 94M, CTC) : le seul de la famille Simba qui tienne le
    # temps réel sans GPU. Simba-S (2.3B) et Simba-W (1.5B) sont des seq2seq
    # autoregressifs — plusieurs dizaines de secondes pour 5 s d'audio sur
    # 8 vCPU. Simba-M (MMS-1b, CTC aussi) est l'alternative plus précise et plus
    # lente : ASR_MODEL=UBC-NLP/Simba-M suffit à basculer, même interface.
    asr_model: str = "UBC-NLP/Simba-H"
    # Non vide => chargé comme adaptateur de langue MMS. Requis par Simba-M, qui
    # livre ses poids de tête dans adapter.<nom>.safetensors et non dans le
    # checkpoint principal : sans ça il transcrit n'importe quoi.
    asr_adapter: str = ""
    tts_model: str = "UBC-NLP/Simba-TTS-lin"

    # --- Audio ---
    # 16 kHz : c'est ce qu'attendent Simba-H ET Simba-TTS-lin. Aucun
    # rééchantillonnage dans toute la chaîne.
    sample_rate: int = 16000
    vad_aggressiveness: int = 2  # 0 = permissif, 3 = agressif
    silence_ms: int = 700  # silence qui clôt un tour de parole
    min_speech_ms: int = 300  # en dessous, c'est un bruit, pas une phrase
    max_turn_s: int = 30  # garde-fou : un micro ouvert ne bloque pas un slot

    # --- CPU (VPS 8 vCPU, pas de GPU) ---
    # 2 conversations x 3 threads torch = 6 vCPU de calcul, il reste 2 vCPU pour
    # l'event loop et le réseau. Les tours suivants attendent leur jeton.
    max_concurrent: int = 2
    torch_threads: int = 3

    log_level: str = "INFO"


settings = Settings()

# Un seul jeton de calcul, partagé par le STT et le TTS : c'est le même CPU
# qu'ils se disputent. Un asyncio.Semaphore ne se lie pas à une boucle à la
# création (Python >= 3.10), on peut donc l'instancier ici.
cpu_slot = asyncio.Semaphore(settings.max_concurrent)
