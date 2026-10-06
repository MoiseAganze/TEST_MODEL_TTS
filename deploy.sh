#!/usr/bin/env bash
#
# Déploiement sur le VPS. Idempotent : relançable autant de fois qu'on veut,
# il ne casse rien et ne laisse jamais le service à moitié démarré.
#
#   ./deploy.sh            build + redémarrage + attente + logs
#   ./deploy.sh --no-build  redémarre sans reconstruire l'image
#
set -euo pipefail
cd "$(dirname "$0")"

CONTAINER=simba_tts
DOMAIN=simba-tts.chronoconcept.net
NETWORK=n8n_default

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
ok()   { printf '    \033[0;32m✓\033[0m %s\n' "$1"; }
die()  { printf '\n\033[1;31m    ✗ %s\033[0m\n\n' "$1" >&2; exit 1; }

if docker compose version >/dev/null 2>&1; then
  DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  DC="docker-compose"
else
  die "Docker Compose est introuvable."
fi

step "Vérification de la configuration"
if [ ! -f .env ]; then
  cp .env.example .env
  die ".env vient d'être créé depuis .env.example.
      Renseigne OPENROUTER_API_KEY dedans, puis relance ./deploy.sh"
fi
grep -qE '^OPENROUTER_API_KEY=.+' .env \
  || die "OPENROUTER_API_KEY est vide dans .env : le LLM ne répondra pas."
docker network inspect "$NETWORK" >/dev/null 2>&1 \
  || die "Le réseau Docker « $NETWORK » n'existe pas. C'est celui de Traefik :
      vérifie que la stack Traefik tourne bien sur ce VPS."
ok ".env présent, clé renseignée, réseau $NETWORK trouvé"

if [ "${1-}" = "--no-build" ]; then
  step "Construction de l'image — ignorée (--no-build)"
else
  step "Construction de l'image"
  # Au premier build, le téléchargement des poids (~500 Mo) prend quelques
  # minutes. Ensuite c'est en cache tant que requirements.txt ne bouge pas.
  $DC build
  ok "image construite"
fi

step "Redémarrage du conteneur"
# `up -d` arrête et remplace l'ancien conteneur de lui-même : pas de stop/rm
# manuels à écrire, et si rien n'a changé il n'y touche pas.
$DC up -d --remove-orphans
ok "conteneur lancé"

step "Attente du démarrage (chargement des modèles, 1 à 2 min)"
deadline=$(( SECONDS + 300 ))
while :; do
  running=$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)
  [ "$running" = true ] || { echo; $DC logs --tail=40; die "Le conteneur s'est arrêté."; }

  case "$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null)" in
    healthy)   echo; ok "conteneur sain"; break ;;
    unhealthy) echo; $DC logs --tail=40; die "Le conteneur est marqué unhealthy." ;;
  esac

  if (( SECONDS >= deadline )); then
    echo; $DC logs --tail=40
    die "Toujours pas sain après 5 min. Les logs ci-dessus disent pourquoi."
  fi
  printf '.'
  sleep 5
done

step "Logs de démarrage"
$DC logs --tail=30

step "Vérification de l'API"
docker exec "$CONTAINER" curl -fsS http://127.0.0.1:8000/health
echo

step "Déployé"
echo "    Page de test : https://$DOMAIN"
echo "    WebSocket    : wss://$DOMAIN/ws/conversation"
echo "    Logs en direct : $DC logs -f"
