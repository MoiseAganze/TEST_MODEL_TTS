# Simba Lingala Voice

API de conversation vocale en **lingala**, en temps réel, sur CPU.
Un seul service FastAPI, un seul endpoint WebSocket.

```
micro ──PCM 16 kHz──▶ VAD ──▶ Simba-H (STT) ──▶ GPT-4o ──▶ Simba-TTS-lin ──PCM──▶ haut-parleur
                   webrtcvad    CTC, 94M       OpenRouter      VITS         phrase par phrase
```

## Les trois décisions qui expliquent le reste

**STT : `UBC-NLP/Simba-H`** (AfriHuBERT, 94M, CTC). La famille Simba compte
cinq modèles ASR ; les trois plus précis (Simba-S 2,3 Md, Simba-W 1,5 Md,
Simba-X 1 Md) sont des seq2seq autoregressifs qui demandent plusieurs dizaines
de secondes pour 5 s d'audio sur 8 vCPU. Simba-H est un CTC : un seul passage,
un argmax, pas de décodage token par token. `ASR_MODEL` permet de basculer vers
`UBC-NLP/Simba-M` (MMS-1b, CTC aussi, plus précis et ~10× plus lent) sans
toucher au code — voir les commentaires de `.env.example`.

**LLM : un seul appel, en streaming.** Pas de chaîne lingala→français,
français→réponse, français→lingala : trois allers-retours OpenRouter ajoutent 3
à 6 s et interdisent le streaming. Le prompt système demande à GPT-4o de
raisonner en français et de ne répondre qu'en lingala, ce qui donne le même
bénéfice en un aller-retour. Les phrases sont découpées à la volée dans le flux
de tokens : la première part au TTS pendant que la suite se génère.

**Transport : PCM brut 16 kHz mono 16 bits, dans les deux sens.** Pas de
WebM/Opus, donc pas de ffmpeg dans l'image et pas de décodage par chunk. 16 kHz
est exactement ce qu'attendent Simba-H et Simba-TTS-lin : zéro
rééchantillonnage dans toute la chaîne.

## Déploiement

```bash
cp .env.example .env     # puis renseigner OPENROUTER_API_KEY
./deploy.sh
```

`deploy.sh` vérifie la config, construit l'image, remplace le conteneur, attend
qu'il soit sain et affiche les logs. Il est relançable sans rien casser.
`./deploy.sh --no-build` redémarre sans reconstruire.

Prérequis sur le VPS : Docker, et la stack Traefik qui fournit le réseau
`n8n_default` — c'est elle qui termine le TLS pour
`simba-tts.chronoconcept.net`.

Le premier build télécharge ~500 Mo de poids, qui sont **gravés dans l'image** :
ensuite le conteneur démarre sans contacter Hugging Face. Compter 60 à 90 s de
chargement des modèles à chaque démarrage, d'où le `start-period` de 180 s du
healthcheck.

## Tester

**Au micro, dans le navigateur** — ouvrir <https://simba-tts.chronoconcept.net>.
La page `static/index.html` capte le micro via un AudioWorklet, envoie du PCM
16 kHz et joue les réponses à la file. HTTPS est obligatoire : aucun navigateur
ne donne le micro en clair.

**En ligne de commande**, avec un wav :

```bash
ffmpeg -i enregistrement.m4a -ac 1 -ar 16000 -sample_fmt s16 question.wav
python test_client.py question.wav wss://simba-tts.chronoconcept.net
```

La réponse vocale est écrite dans `reply.wav`.

**Santé du service** :

```bash
curl -s https://simba-tts.chronoconcept.net/health
```

`slots_libres` y indique combien de conversations peuvent encore être traitées
sans attendre.

## Protocole `/ws/conversation`

| Client → serveur | |
|---|---|
| trame binaire | chunk de micro, PCM 16 kHz mono 16 bits, taille libre |
| `{"type":"reset"}` | oublie l'historique de la conversation |

| Serveur → client | |
|---|---|
| `{"type":"listening"}` | fin de parole détectée, le pipeline démarre |
| `{"type":"transcript","text":…}` | ce qui a été entendu |
| `{"type":"reply","text":…}` | une phrase de la réponse, juste avant son audio |
| trame binaire | le PCM de cette phrase |
| `{"type":"turn_end","ms":…}` | fin du tour et latence totale |
| `{"type":"error","message":…}` | |

C'est la VAD serveur qui décide de la fin d'un tour : `SILENCE_MS` de silence
après au moins `MIN_SPEECH_MS` de parole. Pendant que l'assistant parle, l'audio
entrant est **jeté** — sans ça le micro du client lui fait transcrire sa propre
voix.

## Fichiers

| | |
|---|---|
| `main.py` | lifespan (chargement des modèles), `/ws/conversation`, `/health` |
| `stt.py` | VAD (découpage des tours) + transcription CTC |
| `llm.py` | OpenRouter en streaming, découpage en phrases |
| `tts.py` | normalisation du texte + VITS phrase par phrase |
| `config.py` | configuration et jeton de concurrence CPU |
| `static/index.html` | page de test au micro |
| `test_client.py` | client de test en ligne de commande |

`tts.py`, `llm.py` et `stt.py` embarquent chacun un self-check sans réseau ni
modèle, que le build exécute :

```bash
python tts.py && python llm.py && python stt.py
```

## Ce qui est volontairement absent

- **Pas de fallback faster-whisper.** Une seconde chaîne STT à maintenir pour un
  cas qui n'arrive pas. Si Simba-H déçoit, `ASR_MODEL=UBC-NLP/Simba-M` existe.
- **Pas de Redis ni de Celery.** La file d'attente est un `asyncio.Semaphore` :
  un seul conteneur, un seul processus, c'est suffisant.
- **Pas de multi-locuteur.** Simba-TTS-lin est mono-voix (`num_speakers: 1`).
- **Pas de reprise de parole (barge-in).** Couper l'assistant en parlant demande
  de détecter la voix pendant la lecture, donc de l'annulation d'écho côté
  serveur. À faire si l'usage le réclame.

## Le piège du TTS, pour la prochaine fois

Le vocabulaire de Simba-TTS-lin fait **39 signes** : `a-z` **sans le x**, les
chiffres, l'espace, l'apostrophe et le tiret. Ni accents, ni `ɛ`/`ɔ`, ni
ponctuation. Or le lingala s'écrit avec `ɛ`, `ɔ` et des tons (*bísó*, *kokóma*),
et c'est précisément ce que produit GPT-4o.

C'est la vraie cause des `RuntimeError: narrow(): length must be non-negative` :
le tokenizer jette tout ce qui sort du vocabulaire, une phrase de ponctuation
devient vide, et VITS plante sur une séquence de longueur nulle. `tts.normalize()`
traite la cause (substitutions `ɛ→e`, `ɔ→o`, `x→ks`, suppression des tons par
décomposition NFD, chiffres épelés en lingala) ; le seuil de 4 caractères et le
`try/except` par phrase restent comme filets. À noter aussi : `2` est le
`pad_token` du modèle, un nombre écrit en chiffres serait donc prononcé comme du
silence.
