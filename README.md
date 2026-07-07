# ISIALAB Annotation Interface

Application web Flask pour l'annotation VLDBench avec **cascade automatique** :
**DeBERTa-v3** → **LLM local (Qwen 7B)** → revue humaine.

Fonctionne sur **Windows, macOS et Linux** via un setup unifié.

## Démarrage rapide

| OS | Commande |
|----|----------|
| Mac / Linux | `./start.sh` |
| Windows PowerShell | `powershell -ExecutionPolicy Bypass -File .\start.ps1` |
| Windows CMD | `start.bat` |

Ouvrir http://127.0.0.1:5000

## Prérequis

1. **Python 3.9+** — https://www.python.org/downloads/
2. **Ollama** — https://ollama.com/

Le script de démarrage installe tout le reste (venv, PyTorch, modèles ML, LLM Ollama).

## Modèles ML (DeBERTa, SBERT)

Les poids fine-tunés (~1,5 Go) ne sont **pas** dans Git.
Au premier lancement, `scripts/setup.py` les télécharge depuis la [Release GitHub v1.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0).

Sans release publiée, définir :
```bash
export MODELS_DOWNLOAD_URL=https://votre-hébergeur/vldbench-models-v1.tar.gz
```

## LLM local

Modèle utilisé : **Qwen2.5-7B** (`qwen2.5:7b-instruct`) — ≥10 Go RAM recommandés.

Forcer un autre modèle Ollama : `OLLAMA_LLM_MODEL=mon-modele:tag ./start.sh`

## Utilisation

1. Uploader un fichier JSON d'annotation
2. Cliquer **Auto-annotate** — la cascade remplit `related` et `similarity_annotation`
3. Corriger manuellement les paires flaggées « human review »
4. Télécharger le JSON annoté

Protocole : `protocole.md`

## API statut

```
GET /api/status
```

## Documentation complète

Voir **[DEPLOYMENT.md](DEPLOYMENT.md)** pour Docker, publication des modèles, et dépannage.

## Dépôt associé

[AI_annotation](https://github.com/Cespriet/AI_annotation) — entraînement, évaluation, cascade CLI.
