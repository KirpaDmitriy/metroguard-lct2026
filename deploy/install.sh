#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "run as root: sudo deploy/install.sh" >&2
    exit 1
fi

DEPLOY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP_DIR=/opt/metroguard
ENV_DIR=/etc/metroguard
DATA_DIR=/var/lib/metroguard/jobs

if [[ ! -f "$APP_DIR/demo_app/requirements.txt" ]]; then
    echo "copy the release to $APP_DIR before running this script" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends curl nginx python3 python3-venv

id -u metroguard >/dev/null 2>&1 || useradd --system --home-dir /nonexistent --shell /usr/sbin/nologin metroguard
install -d -o metroguard -g metroguard -m 0750 "$DATA_DIR"
install -d -o root -g metroguard -m 0750 "$ENV_DIR"
install -d -o root -g root -m 0755 /var/www/certbot

CERT_DIR=/etc/letsencrypt/live/93-77-179-211.nip.io
if [[ ! -r "$CERT_DIR/fullchain.pem" || ! -r "$CERT_DIR/privkey.pem" ]]; then
    echo "Let's Encrypt certificate is missing from $CERT_DIR" >&2
    exit 1
fi

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/python" -m pip install --disable-pip-version-check --no-cache-dir -r "$APP_DIR/demo_app/requirements.txt"
chown -R root:root "$APP_DIR"
chmod -R u=rwX,go=rX "$APP_DIR"

if [[ ! -f "$ENV_DIR/demo.env" ]]; then
    install -o root -g metroguard -m 0640 "$DEPLOY_DIR/demo.env.example" "$ENV_DIR/demo.env"
fi
install -o root -g root -m 0644 "$DEPLOY_DIR/metroguard.service" /etc/systemd/system/metroguard.service
install -o root -g root -m 0644 "$DEPLOY_DIR/metroguard.nginx" /etc/nginx/sites-available/metroguard
install -o root -g root -m 0644 "$DEPLOY_DIR/metroguard-limits.conf" /etc/nginx/conf.d/metroguard-limits.conf
install -o root -g root -m 0755 "$DEPLOY_DIR/smoke.sh" /usr/local/bin/metroguard-smoke
ln -sfn /etc/nginx/sites-available/metroguard /etc/nginx/sites-enabled/metroguard
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl daemon-reload
systemctl enable --now metroguard nginx
systemctl restart metroguard
systemctl reload nginx
/usr/local/bin/metroguard-smoke http://127.0.0.1:8000
