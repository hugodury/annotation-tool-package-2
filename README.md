# ISIALAB Annotation Interface

Application web Flask pour l'annotation VLDBench avec **cascade automatique** :
**DeBERTa-v3** → **LLM local (Qwen 7B)** → revue humaine.

Fonctionne sur **Windows, macOS et Linux** via un setup unifié (`start.sh` / `start.bat` / `start.ps1`).

**Dépôt :** https://github.com/hugodury/annotation-tool-package-2

```bash
git clone https://github.com/hugodury/annotation-tool-package-2.git
cd annotation-tool-package-2
```

> **1er lancement** : 15–45 min selon la machine (téléchargements + installation). Internet requis une seule fois.

## Démarrage rapide

| OS | Commande |
|----|----------|
| Mac / Linux | `./start.sh` |
| Windows PowerShell | `powershell -ExecutionPolicy Bypass -File .\start.ps1` |
| Windows CMD | `start.bat` |

Ouvrir http://127.0.0.1:5000

## Prérequis

| Élément | Détail |
|---------|--------|
| Python | 3.9+ (3.12+ recommandé) — [python.org](https://www.python.org/downloads/) |
| Ollama | [ollama.com](https://ollama.com/) |
| RAM | 10 Go recommandés (Qwen 7B) |
| Disque (1er lancement) | **~13 Go libres** (venv, modèles ML, LLM, marge) |
| Disque (une fois installé) | **~8 Go** au total |

Le script de démarrage installe le reste : venv, PyTorch (CPU / CUDA / MPS), modèles ML, pull Qwen via Ollama.

## Checklist Configuration (interface web)

Au chargement de l'app, une checklist vérifie **7 prérequis obligatoires** :

- Python, PyTorch, sentence-transformers ≥ 5.5
- Modèles ML (DeBERTa, SBERT, cross-encoder)
- Ollama installé et actif
- LLM `qwen2.5:7b-instruct` téléchargé

**Run Model** n'est disponible que si tous ces points sont ✓ (bouton désactivé + blocage API sinon).
RAM, GPU et performance sont informatifs seulement.

### Routage cascade (résumé)

| Situation | Comportement |
|-----------|--------------|
| DeBERTa confiant (≥ 95 %) | Annotation automatique |
| Classe `against` / `not_related` ambiguë | DeBERTa seul |
| Classe `supporting` / `undetermined` ambiguë | LLM Qwen (P3) puis consensus ou revue humaine |
| Désaccord fort DeBERTa / LLM | Rejet ou flag « human review » |
| Timeout LLM | Revue humaine, le batch continue |

## Modèles ML (DeBERTa, SBERT)

Les poids fine-tunés (~1,5 Go) ne sont **pas** dans Git.
Au premier lancement, `scripts/setup.py` les télécharge depuis la [Release GitHub v1.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0).

Sans release publiée :
```bash
export MODELS_DOWNLOAD_URL=https://votre-hébergeur/vldbench-models-v1.tar.gz
```

## LLM local

Modèle : **Qwen2.5-7B** (`qwen2.5:7b-instruct`).

Forcer un autre modèle : `OLLAMA_LLM_MODEL=mon-modele:tag ./start.sh`

Variables optionnelles : copier `.env.example` vers `.env` (`OLLAMA_HOST`, `OLLAMA_LLM_MODEL`, `MODELS_DOWNLOAD_URL`, etc.).

### Timeouts et keep_alive

Configurés dans `cascade/config.json` :

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `inference.timeout` | 600 s (10 min) | Temps max par appel LLM |
| `run_model.no_annotation_timeout` | 360 s (6 min) | Arrêt si aucune cible pré-remplie |
| `inference.keep_alive` | 30 min | Qwen reste en RAM après le dernier appel LLM (appels suivants plus rapides) |

`keep_alive` n'est pas une limite de session : chaque appel LLM renouvelle le délai.

## Utilisation

1. Uploader un fichier JSON d'annotation
2. Vérifier la checklist **Configuration** (badge « Tout installé »)
3. Cliquer **Run Model** — la cascade remplit `related` et `similarity_annotation`
4. Corriger manuellement les paires en revue humaine
5. Télécharger le JSON annoté

Protocole : `protocole.md`

**Session** : *Resume Last Session* reprend le dernier fichier ; re-uploader un JSON déjà annoté conserve les statuts.

## Dépannage rapide

```bash
# Vérifier la machine (sans lancer l'app)
python scripts/system_check.py

# Relancer le setup complet
./start.sh          # Mac / Linux
start.bat           # Windows CMD
```

Si Run Model s'arrête en cours de route, la progression partielle est sauvegardée (JSON + base locale).
Un chronomètre affiche la durée totale d'annotation à la fin du batch.

## API statut

```
GET /api/status
```

Retourne la checklist, `ready_for_run_model`, espace disque recommandé, état Ollama, etc.

## Documentation complète

Voir **[DEPLOYMENT.md](DEPLOYMENT.md)** pour Docker, publication des modèles, et dépannage.

## Dépôt associé

[AI_annotation](https://github.com/Cespriet/AI_annotation) — entraînement, évaluation, cascade CLI.
