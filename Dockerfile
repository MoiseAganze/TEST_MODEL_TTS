FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Les poids sont téléchargés AU BUILD dans ce dossier (voir plus bas). Sans
    # HF_HOME ici, le cache atterrirait dans /root/.cache et serait réécrit à
    # chaque démarrage de conteneur.
    HF_HOME=/app/.hf \
    # Sans ça, transformers avertit à chaque fork d'uvicorn que le parallélisme
    # des tokenizers est désactivé — trois lignes de bruit dans tous les logs.
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# NOTE: Ne mets pas `git pull` dans un Dockerfile (image non reproductible +
# credentials). Pour déployer : pull sur l'hôte puis rebuild/restart (deploy.sh).

# curl : utilisé par le HEALTHCHECK plus bas, absent de l'image slim.
#
# Pas de ffmpeg : le WebSocket échange du PCM 16 kHz brut dans les deux sens,
#   il n'y a rien à décoder. Si un jour le client envoie du WebM/Opus il faudra
#   l'ajouter, et pas avant : ça pèse 200 Mo.
# Pas de build-essential : requirements.txt prend webrtcvad-wheels, qui fournit
#   des roues précompilées, là où le paquet webrtcvad d'origine n'est publié
#   qu'en source et réclamerait gcc.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Les poids descendent AVANT le code : une modification de main.py ne doit pas
# relancer 500 Mo de téléchargement.
#
# C'est le pendant du `pnpm build` : le travail lourd au build, pas à
# l'exécution. Le conteneur démarre ensuite sans contacter Hugging Face, et un
# restart ne retélécharge rien.
#
# Ces ARG servent aussi d'ENV à l'exécution (config.py lit ASR_MODEL/TTS_MODEL) :
# une seule source de vérité. Le .env, chargé par compose, les écrase si besoin —
# mais un modèle changé dans .env ne sera pas préchargé dans l'image, donc le
# premier démarrage sera long.
ARG ASR_MODEL=UBC-NLP/Simba-H
ARG TTS_MODEL=UBC-NLP/Simba-TTS-lin
ENV ASR_MODEL=$ASR_MODEL \
    TTS_MODEL=$TTS_MODEL
RUN python -c "import os; from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, VitsModel; a=os.environ['ASR_MODEL']; t=os.environ['TTS_MODEL']; AutoProcessor.from_pretrained(a); AutoModelForCTC.from_pretrained(a); AutoTokenizer.from_pretrained(t); VitsModel.from_pretrained(t); print(f'poids en cache : {a} + {t}')"

COPY . .

# Les trois modules qui portent de la logique non triviale embarquent un
# self-check sans réseau ni modèle. Les faire tourner au build coûte deux
# secondes et refuse une image dont la VAD, le découpage en phrases ou la
# normalisation du texte sont cassés.
RUN python tts.py && python llm.py && python stt.py

EXPOSE 8000

# start-period long : charger les poids torch en CPU prend 60 à 90 s. Avec les
# 40 s du projet d'origine, Docker tuerait le conteneur en boucle avant qu'il
# ait fini de démarrer.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Un seul worker, volontairement : chaque worker chargerait sa propre copie des
# modèles (500 Mo de RAM chacun) et ils se disputeraient les mêmes vCPU. La
# concurrence est gérée dans l'application, par MAX_CONCURRENT.
#
# uvicorn envoie de lui-même un ping WebSocket toutes les 20 s, ce qui tient la
# connexion ouverte à travers Traefik sans ping applicatif.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
