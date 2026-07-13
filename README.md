# ISIALAB Annotation Interface

Application web Flask pour l'annotation VLDBench avec **cascade automatique** :
**DeBERTa-v3** → **LLM local (Qwen 7B via Ollama)** → revue humaine.

Fonctionne sur **Windows, macOS et Linux** via un setup unifié (`start.sh` / `start.bat` / `start.ps1`), ou via **Docker**.

| Ressource | Lien |
|-----------|------|
| **Dépôt GitHub** | https://github.com/hugodury/annotation-tool-package-2 |
| **Release modèles ML (v1.0.0)** | https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0 |

```bash
git clone https://github.com/hugodury/annotation-tool-package-2.git
cd annotation-tool-package-2
```

> **1er lancement** : 15–45 min selon la machine (téléchargements + installation). Internet requis une seule fois.

---

## Démarrage rapide

| OS | Commande |
|----|----------|
| Mac / Linux | `./start.sh` |
| Windows PowerShell | `powershell -ExecutionPolicy Bypass -File .\start.ps1` |
| Windows CMD | `start.bat` |
| Docker | `docker compose up --build` (voir [DEPLOYMENT.md](DEPLOYMENT.md)) |

Ouvrir **http://127.0.0.1:5000**

---

## Prérequis

| Élément | Détail |
|---------|--------|
| Python | 3.9+ (3.12+ recommandé) — [python.org](https://www.python.org/downloads/) |
| Ollama | [ollama.com](https://ollama.com/) |
| RAM | 10 Go recommandés (Qwen 7B) |
| Disque (1er lancement) | **~13 Go libres** (venv, modèles ML ~1,5 Go, LLM ~4 Go, marge) |
| Disque (une fois installé) | **~8 Go** au total |

Le script de démarrage installe le reste : venv, PyTorch (CPU / CUDA / MPS), modèles ML, pull Qwen via Ollama.

> **Note disque** : l'espace libre affiché sur votre machine (ex. 20 Go) doit rester **au-dessus** des ~13 Go requis au premier lancement — ce ne sont pas la même chose.

---

## Checklist Configuration (interface web)

Au chargement de l'app, une checklist vérifie **7 prérequis obligatoires** :

- Python, PyTorch, sentence-transformers ≥ 5.5
- Modèles ML (DeBERTa, SBERT, cross-encoder)
- Ollama installé et actif
- LLM `qwen2.5:7b-instruct` téléchargé

**Run Model** n'est disponible que si tous ces points sont ✓ (bouton désactivé + blocage API sinon).
RAM, GPU et performance sont informatifs seulement.

Ollama et le LLM se préparent en arrière-plan au chargement de la page (`/api/ensure-ollama`).

---

## Routage cascade

| Situation | Route | Comportement |
|-----------|-------|--------------|
| DeBERTa confiant (≥ τ, défaut 95 %) | `deberta_auto` | Annotation automatique + similarité SBERT |
| Classe `against` / `not_related` ambiguë | `deberta_ambiguous` | DeBERTa seul |
| Classe `supporting` / `undetermined` ambiguë | `consensus` ou `human` | LLM Qwen (prompt P3) puis consensus ou revue |
| Désaccord fort DeBERTa / LLM (conf. ≥ 85 %) | `rejected` | Rejet — revue humaine |
| Désaccord modéré | `human` | Flag revue humaine |
| Timeout / erreur LLM | `human` | Revue humaine, le batch continue |

Seuil τ réglable dans l'interface (champ **Seuil τ**, défaut `0.95`).

---

## Modèles ML (DeBERTa, SBERT, cross-encoder)

Les poids fine-tunés (~1,5 Go) ne sont **pas** dans Git.

Au premier lancement, `scripts/setup.py` les télécharge depuis la
[Release GitHub v1.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0)
(archive `vldbench-models-v1.tar.gz`).

Sans release accessible :
```bash
export MODELS_DOWNLOAD_URL=https://votre-hébergeur/vldbench-models-v1.tar.gz
```

URL configurée dans `models.manifest.json`.

---

## LLM local

Modèle par défaut : **Qwen2.5-7B** (`qwen2.5:7b-instruct`).

Forcer un autre modèle :
```bash
OLLAMA_LLM_MODEL=mon-modele:tag ./start.sh
```

Variables optionnelles : copier `.env.example` vers `.env` (`OLLAMA_HOST`, `OLLAMA_LLM_MODEL`, `MODELS_DOWNLOAD_URL`, etc.).

### Timeouts, retries et mode CPU lent

Configurés dans `cascade/config.json` :

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `inference.timeout` | 600 s (10 min) | Temps max par appel LLM |
| `inference.llm_retries` | 2 | Nouvelles tentatives si Ollama échoue |
| `inference.keep_alive` | 30 min | Qwen reste en RAM entre les appels |
| `run_model.no_annotation_timeout` | 360 s | Arrêt si aucune cible annotée |
| `run_model.save_every_n_refs` | 3 | Sauvegarde JSON + DB tous les N refs |
| `run_model.cpu_slow.inference_timeout_sec` | 900 s | Timeout LLM allongé sur CPU |
| `run_model.cpu_slow.max_refs_suggested` | 50 | Alerte si batch trop large sur CPU |
| `run_model.cpu_slow.sec_per_target_estimate` | 5 s | Estimation durée par cible (CPU) |

Sur CPU, les cibles passant par le LLM peuvent prendre **30 s à 2 min** chacune — la barre de progression affiche « en cours » pendant ce délai.

---

## Utilisation

### 1. Charger un corpus

- **Charger** : upload d'un fichier JSON VLDBench
- **Reprendre la dernière session** : dernier fichier dans `uploads/`
- **Sessions enregistrées** : liste déroulante de tous les JSON présents dans `uploads/`
- **Fichier actif** : affiché en haut du panneau de gauche

### 2. Vérifier la configuration

Badge **Tout installé** requis avant **Run Model**.

### 3. Run Model (annotation automatique)

1. Indiquer **index début** et **index fin** (plage de références)
2. Optionnel : ajuster le **seuil τ** (défaut 0,95)
3. Cliquer **Run Model** — le batch s'exécute **en arrière-plan** (réponse HTTP 202)
4. Suivre la **barre de progression** : cibles, références, message d'étape
5. **Continuer** : reprend à la prochaine référence incomplète de la plage
6. **Annuler** : arrêt du batch en cours (voir ci-dessous)
7. **Voir les logs** : tail du fichier `logs/run_model_YYYYMMDD.log`

Estimation de durée affichée avant lancement (`/api/auto_annotate/estimate`).

À la fin : **résumé en français** (DeBERTa auto, consensus, revue humaine, durée, lien vers la première référence à corriger).

#### Annulation pendant Run Model

Le bouton **Annuler** (overlay de progression) :

1. Envoie `POST /api/auto_annotate/cancel` — l'interface affiche « Annulation demandée… »
2. **Interrompt** l'appel LLM en cours (Ollama en streaming) ou attend la fin de l'inférence DeBERTa (~1 s max)
3. Fonctionne aussi pendant le **chargement des modèles** (avant la première cible)
4. **Sauvegarde** les cibles déjà traitées dans le JSON + checkpoint SQLite
5. Affiche un **résumé partiel** et ferme l'overlay
6. La cible en cours au moment du clic n'est en général **pas** enregistrée

Pour reprendre : **Continuer** puis **Run Model** (reprise à la prochaine référence incomplète).

---

### 4. Annotation manuelle

- Naviguer entre les références (Précédent / Suivant / liste)
- **Filtre** : toutes, en attente, partielles, terminées, à revoir
- **Prochaine revue** : saute à la prochaine référence avec désaccord ou flag humain
- **Compteur corpus** : stats globales (complete / partielle / en attente, %)
- **Légende cascade** : badges DeBERTa auto, consensus LLM, revue humaine
- **Enregistrer et suivant** : sauvegarde puis saut à la prochaine référence incomplète
- **Ignorer et suivant** : dismiss toutes les cibles de la référence

### 5. Exporter

- **Télécharger JSON** : fichier annoté depuis `uploads/`
- **Resync DB** : resynchronise le cache SQLite depuis le JSON (source de vérité)
- **Vider le cache SQLite** : efface l'historique local **sans** supprimer les JSON

Protocole détaillé : `protocole.md`

---

## Où sont stockées les données ?

| Emplacement | Rôle |
|-------------|------|
| `uploads/*.json` | **Source de vérité** — annotations, routes cascade, similarités |
| `uploads/backups/` | Sauvegardes automatiques avant chaque Run Model |
| `instance/annotations.db` | Cache SQLite (statuts UI, reprise) — régénérable via Resync |
| `logs/run_model_*.log` | Journaux des batches Run Model |

Les champs écrits par la cascade dans chaque cible : `related`, `similarity_annotation`, `cascade_route`, `model_confidence` (+ `llm_pred` / `llm_error` si applicable).

---

## Robustesse

- **Verrous fichier** : écritures JSON synchronisées (`scripts/annotation_store.py`)
- **Sauvegarde incrémentale** : checkpoint JSON + DB tous les 3 refs (configurable)
- **Reprise partielle** : skip des refs complètes et des cibles déjà annotées
- **Backup auto** avant chaque batch
- **Validation** : indices de plage, taille d'annotation à l'enregistrement manuel
- **Parsing LLM robuste** : extraction JSON imbriquée + blocs ` ```json ` + retries Ollama
- **Ollama en streaming** : permet l'annulation rapide des appels LLM en cours
- **Annulation interruptible** : DeBERTa, LLM et chargement des modèles
- **Progression temps réel** : polling `/api/auto_annotate/status` toutes les 2 s (barre « en cours »)
- **Reprise au rechargement** : si un batch était en cours, l'overlay reprend automatiquement

---

## API

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Interface web |
| GET | `/api/health` | Santé minimale (Docker healthcheck) |
| GET | `/api/status` | Checklist configuration complète |
| POST | `/api/ensure-ollama` | Prépare Ollama + pull LLM en arrière-plan |
| GET | `/api/system-check` | Alias diagnostic |
| POST | `/upload` | Upload JSON |
| GET | `/resume` | Reprend la dernière session |
| GET | `/download` | Télécharge le JSON actif |
| POST | `/save_annotation` | Sauvegarde manuelle d'une référence |
| POST | `/clear_database` | Vide le cache SQLite |
| GET | `/api/sessions` | Liste les fichiers JSON dans `uploads/` |
| POST | `/api/sessions/load` | Charge un fichier de session |
| POST | `/api/resync` | Resync DB ← JSON |
| GET | `/api/logs/run_model/latest` | Dernières lignes du log Run Model |
| POST | `/auto_annotate` | Lance Run Model (async, 202) |
| GET | `/api/auto_annotate/status` | État et progression du batch |
| POST | `/api/auto_annotate/estimate` | Estimation durée / cibles |
| GET | `/api/auto_annotate/resume` | Index de reprise dans une plage |
| POST | `/api/auto_annotate/cancel` | Annulation interruptible du batch en cours |

---

## Docker

```bash
docker compose up --build
```

Services : `app` (Flask, port 5000) + `ollama` (port 11434).
Volumes montés : `./models`, `./uploads`, `./logs`.
Healthchecks sur les deux services.

Voir **[DEPLOYMENT.md](DEPLOYMENT.md)** pour les détails.

---

## Dépannage rapide

```bash
# Vérifier la machine (sans lancer l'app)
python scripts/system_check.py

# Vérifier l'espace disque
python scripts/disk_check.py

# Relancer le setup complet
./start.sh          # Mac / Linux
start.bat           # Windows CMD
```

| Problème | Solution |
|----------|----------|
| Progression bloquée à 0/0 | Redémarrer Flask (`./start.sh`) — ancien serveur sans les routes API |
| Compteur lent sur 0/N | Normal pendant un appel LLM (30 s–2 min/cible) ; la barre affiche « en cours » |
| Run Model indisponible | Compléter la checklist (modèles ML, Ollama, Qwen) |
| Session perdue après reload | **Reprendre la dernière session** ou sélecteur **Sessions** |
| DB désynchronisée | **Resync DB** depuis l'interface |
| Annuler sans effet | Recharger la page (Ctrl+F5) — l'annulation interrompt le LLM en streaming sous ~10 s |
| Batch interrompu | Progression partielle sauvegardée ; **Continuer** pour reprendre |

---

## Structure du projet

```
annotation-tool-package-2/
├── app.py                    # Flask, routes, modèles SQLAlchemy
├── cascade/                  # Moteur cascade (DeBERTa + LLM)
├── scripts/
│   ├── setup.py              # Installation multi-OS
│   ├── run_model_job.py      # Batch async Run Model
│   ├── annotation_store.py   # Verrous JSON, backups
│   ├── system_check.py       # Diagnostic machine
│   └── ollama_service.py     # Gestion Ollama
├── templates/index.html      # Interface web
├── static/app_extras.js      # Toasts, sessions, filtres
├── uploads/                  # JSON annotés (non versionnés)
├── logs/                     # Logs Run Model (non versionnés)
├── instance/                 # SQLite cache (non versionné)
├── models/                   # Poids ML (téléchargés via Release)
├── docker-compose.yml
├── Dockerfile
└── start.sh / start.bat / start.ps1
```

---

## Documentation complète

- **[DEPLOYMENT.md](DEPLOYMENT.md)** — Docker, publication des modèles, dépannage avancé
- **Release modèles** — https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0

## Dépôt associé

[AI_annotation](https://github.com/Cespriet/AI_annotation) — entraînement, évaluation, cascade CLI.
