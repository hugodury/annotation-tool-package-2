# ISIALAB Annotation Interface

Application web Flask pour l'annotation VLDBench avec **cascade automatique** :
**DeBERTa-v3** → **LLM local (Qwen / Phi / Llama)** → revue humaine.

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
Au premier lancement, `scripts/setup.py` les télécharge depuis la [Release GitHub v1.0.0](https://github.com/Cespriet/annotation-tool-package-2/releases/tag/v1.0.0).

Sans release publiée, définir :
```bash
export MODELS_DOWNLOAD_URL=https://votre-hébergeur/vldbench-models-v1.tar.gz
```

## LLM local

Sélection automatique selon la RAM :
- **Qwen2.5-7B** — machines avec ≥10 Go RAM
- **Phi-3.5-3.8B** ou **Llama3.2-3B** — PC modestes (≥6 Go)

Forcer un modèle : `OLLAMA_LLM_MODEL=phi3.5:3.8b ./start.sh`

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
