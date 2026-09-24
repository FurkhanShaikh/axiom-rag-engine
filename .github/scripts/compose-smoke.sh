#!/usr/bin/env bash
# Start the whole docker-compose stack and check every service does its job,
# so image bumps to any of them (Dependabot included) are tested by CI rather
# than by hand. Run from the repository root.
set -euo pipefail

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    docker compose ps -a || true
    docker compose logs --no-color --tail 200 || true
  fi
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

# compose reads .env; CI has none. Development mode: no API keys, no LLM
# provider and mock search are all allowed, so the stack boots keyless.
[ -f .env ] || printf 'AXIOM_ENV=development\n' > .env

retry() {  # retry <attempts> <command...>: until it succeeds, 2 s apart
  local n=$1; shift
  for _ in $(seq 1 "$n"); do "$@" && return 0; sleep 2; done
  "$@"
}

healthy() {  # healthy <container>: its compose healthcheck reports healthy
  [ "$(docker inspect -f '{{.State.Health.Status}}' "$1" 2>/dev/null)" = "healthy" ]
}

# Not `up --wait`: it can count the one-shot ollama-init step, which exits by
# design, as a failure. The engine starts only after Ollama and Redis are
# healthy (depends_on), so waiting for the engine's healthcheck covers them.
docker compose up -d --build
retry 210 healthy axiom-rag-engine
for c in axiom-ollama axiom-redis; do healthy "$c" || { echo "$c is not healthy"; exit 1; }; done
echo "engine, Ollama and Redis healthy"

echo "== engine: ready, with Redis as its cache"
curl -fsS http://localhost:8000/health/ready > ready.json
curl -fsS http://localhost:8000/v1/status > status.json
python3 - <<'PY'
import json
ready = json.load(open("ready.json"))
cache = json.load(open("status.json"))["cache"]["backend"]
assert ready["status"] == "ok", ready
assert cache == "RedisCacheBackend", f"cache is {cache}, not Redis"
print("engine ok:", ready, "| cache:", cache)
PY

echo "== engine -> Ollama over the compose network"
docker compose exec -T axiom-rag-engine curl -fsS http://ollama:11434/api/tags
echo

echo "== Prometheus: ready and scraping the engine"
retry 30 curl -fsS http://localhost:9090/-/ready
scraped() {
  curl -fsS 'http://localhost:9090/api/v1/query?query=up%7Bjob%3D%22axiom-rag-engine%22%7D' \
    | python3 -c 'import json,sys; r=json.load(sys.stdin)["data"]["result"]; sys.exit(0 if r and r[0]["value"][1]=="1" else 1)'
}
retry 45 scraped
echo "prometheus ok: axiom-rag-engine target is up"

echo "== Grafana: healthy, dashboard provisioned, Prometheus datasource working"
auth=admin:admin  # compose's documented default login
retry 30 curl -fsS http://localhost:3000/api/health
echo
dashboards=$(curl -fsS -u "$auth" 'http://localhost:3000/api/search?query=Axiom%20Engine')
python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert any(x.get("title","").startswith("Axiom Engine") for x in d), d; print("dashboard ok:", [x["title"] for x in d])' "$dashboards"
retry 15 curl -fsS -u "$auth" http://localhost:3000/api/datasources/uid/prometheus/health
echo
echo "compose stack ok"
