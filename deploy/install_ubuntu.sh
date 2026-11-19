#!/usr/bin/env bash
# One-time setup on a free Ubuntu cloud VM (e.g. Oracle Cloud Always Free). Run from the project folder:
#   bash deploy/install_ubuntu.sh
set -euo pipefail
sudo timedatectl set-timezone Asia/Kolkata
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip debian-keyring debian-archive-keyring apt-transport-https curl
python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
# Caddy = automatic HTTPS for nifty.ujesh.in
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy
sed "s#/home/ubuntu/niftysim#$(pwd)#g; s#User=ubuntu#User=$(whoami)#" deploy/niftysim.service | sudo tee /etc/systemd/system/niftysim.service
sudo systemctl daemon-reload && sudo systemctl enable --now niftysim
echo "Done. (Optional: add NIFTYSIM_PASSWORD=... to .env, then: sudo systemctl restart niftysim)"
echo "Point an A record for nifty.ujesh.in at this server's public IP and open ports 80/443."
