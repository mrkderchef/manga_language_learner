#!/usr/bin/env bash
# ============================================================
# Ollama Setup Script – für den Remote GPU-Rechner (10.100.10.112)
# 
# Dieses Script auf dem Remote-Rechner ausführen:
#   ssh user@10.100.10.112
#   bash setup_ollama_remote.sh
# ============================================================
set -euo pipefail

echo "=== Ollama Setup für Manga Language Learner ==="
echo ""

# 1. Ollama installieren (falls noch nicht vorhanden)
if command -v ollama &> /dev/null; then
    echo "[OK] Ollama ist bereits installiert: $(ollama --version)"
else
    echo "[*] Installiere Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "[OK] Ollama installiert"
fi

# 2. Ollama-Service starten (lauscht auf 0.0.0.0:11434)
echo ""
echo "[*] Konfiguriere Ollama für Netzwerk-Zugriff..."
# Setze OLLAMA_HOST damit Ollama auf allen Interfaces lauscht
export OLLAMA_HOST=0.0.0.0:11434

# Systemd override für persistente Konfiguration
if systemctl is-active --quiet ollama 2>/dev/null; then
    echo "[*] Stoppe laufenden Ollama-Service..."
    sudo systemctl stop ollama
fi

# Erstelle systemd override für OLLAMA_HOST
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF
sudo systemctl daemon-reload

echo "[*] Starte Ollama-Service..."
sudo systemctl enable ollama
sudo systemctl start ollama

# Warten bis Ollama bereit ist
echo "[*] Warte auf Ollama..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[OK] Ollama läuft"
        break
    fi
    sleep 1
done

# 3. Modelle herunterladen
echo ""
echo "[*] Lade empfohlenes Sugoi-Modell für Ollama-Übersetzungen..."
ollama pull hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M

# 4. Verfügbare Modelle anzeigen
echo ""
echo "=== Installierte Modelle ==="
ollama list

# 5. Verbindung testen
echo ""
echo "=== Verbindungstest ==="
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "Ollama API erreichbar unter: http://${LOCAL_IP}:11434"
echo ""
echo "Test von einem anderen Rechner:"
echo "  curl http://${LOCAL_IP}:11434/api/tags"
echo ""
echo "=== Setup abgeschlossen ==="
