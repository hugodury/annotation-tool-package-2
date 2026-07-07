# Déploiement multi-OS — ISIALAB Annotation Interface

## Réponse courte

**Non**, un simple `git clone` ne suffit pas : les poids ML (~1,5 Go) ne sont pas versionnés dans Git.
Avec ce guide + une **GitHub Release** des modèles, l'auto-annotation DeBERTa / Qwen 7B fonctionne sur **Windows, macOS et Linux**.

---

## Prérequis utilisateur final

| Composant | Windows | macOS | Linux |
|-----------|---------|-------|-------|
| Python 3.9+ | ✓ | ✓ | ✓ |
| Ollama | [ollama.com](https://ollama.com/) | idem | idem |
| RAM | 10 Go recommandés (Qwen 7B) | idem | idem |
| Disque | ~12–15 Go (app + modèles ML + LLM) | idem | idem |

---

## Installation utilisateur (1 commande)

### Mac / Linux
```bash
git clone https://github.com/hugodury/annotation-tool-package-2.git
cd annotation-tool-package-2
chmod +x start.sh
./start.sh
```

### Windows (PowerShell)
```powershell
git clone https://github.com/hugodury/annotation-tool-package-2.git
cd annotation-tool-package-2
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Le script `scripts/setup.py` :
1. Crée le venv Python
2. Installe PyTorch (CPU / CUDA / MPS selon la machine)
3. Télécharge les modèles DeBERTa + SBERT + cross-encoder
4. Démarre Ollama si besoin et télécharge Qwen 7B
5. Lance l'app sur http://127.0.0.1:5000

---

## LLM

| Profil | Modèle Ollama | RAM min | Usage |
|--------|---------------|---------|-------|
| qwen7b | `qwen2.5:7b-instruct` | 10 Go | LLM de la cascade |

Forcer un autre modèle Ollama :
```bash
export OLLAMA_LLM_MODEL=qwen2.5:7b-instruct   # Mac/Linux
set OLLAMA_LLM_MODEL=qwen2.5:7b-instruct       # Windows CMD
```

---

## Instructions mainteneur — publier les modèles sur GitHub

Les poids ne vont **pas** dans Git (trop volumineux). Procédure :

```bash
# 1. Placer les modèles dans models/
# 2. Créer l'archive
python scripts/package_models.py
# → vldbench-models-v1.tar.gz (~1,4 Go)

# 3. Créer une release GitHub tag v1.0.0 et y attacher vldbench-models-v1.tar.gz
gh release create v1.0.0 vldbench-models-v1.tar.gz \
  --title "VLDBench models v1" \
  --notes "DeBERTa + SBERT + cross-encoder fine-tuned weights"
```

L'URL est déjà configurée dans `models.manifest.json`.

Alternative si pas de release : héberger l'archive ailleurs et définir :
```bash
export MODELS_DOWNLOAD_URL=https://votre-url/vldbench-models-v1.tar.gz
```

---

## Docker (optionnel)

```bash
# Télécharger les modèles localement d'abord, ou monter MODELS_DOWNLOAD_URL
docker compose up --build
```

- App : http://localhost:5000
- Ollama : http://localhost:11434

Après le premier démarrage, entrer dans le conteneur ollama pour pull le LLM :
```bash
docker compose exec ollama ollama pull qwen2.5:7b-instruct
```

---

## Vérification

```bash
curl http://127.0.0.1:5000/api/status
```

Réponse attendue :
```json
{
  "ml_models": true,
  "ollama": { "installed": true, "running": true },
  "active_llm": { "ollama": "qwen2.5:7b-instruct" }
}
```

---

## Dépannage

| Problème | Solution |
|----------|----------|
| `Modele DeBERTa manquant` | `python scripts/setup.py` |
| `Ollama service error` | `ollama serve` puis `ollama pull qwen2.5:7b-instruct` |
| PyTorch lent | Normal en CPU ; installer CUDA si GPU NVIDIA |
| Download échoue | Publier la Release ou définir `MODELS_DOWNLOAD_URL` |
