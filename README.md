# ISIALAB Annotation Interface

Application web Flask pour l'annotation VLDBench avec **cascade automatique** :
**DeBERTa-v3** → **LLM local (Qwen 7B via Ollama)** → revue humaine.

Interface web **entièrement en anglais**. Fonctionne sur **Windows, macOS et Linux** via un setup unifié (`start.sh` / `start.bat` / `start.ps1`), ou via **Docker**.

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

> Après modification du code HTML/JS/Python, **redémarrer Flask** (`./start.sh`) si le serveur tournait déjà — les templates sont rechargés automatiquement au prochain démarrage (`TEMPLATES_AUTO_RELOAD`).

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

### Sélecteurs système (Browse / Choose JSON)

Les boutons **Browse** (dossier de stockage) et **Choose JSON file** ouvrent le **sélecteur natif** de l'OS, avec repli automatique :

| OS | Dossier | Fichier JSON |
|----|---------|--------------|
| **Linux** | zenity → kdialog → yad → tkinter | idem |
| **macOS** | Finder (osascript) → tkinter | idem |
| **Windows** | PowerShell / pwsh → tkinter | idem |

Sur Linux, une session bureau (`DISPLAY` ou `WAYLAND_DISPLAY`) est requise. L'app vérifie la disponibilité via `GET /api/dialogs/capabilities`.

---

## Checklist configuration (interface web)

Au chargement, une checklist (en anglais) vérifie **7 prérequis obligatoires** :

- Python, PyTorch, sentence-transformers ≥ 5.5
- ML models (DeBERTa, SBERT, cross-encoder)
- Ollama installed & running
- LLM `qwen2.5:7b-instruct` downloaded

**Run Model** n'est disponible que si tous ces points sont ✓ (bouton désactivé + blocage API sinon).
RAM, GPU et performance estimée sont informatifs seulement.

Ollama et le LLM se préparent en arrière-plan (`/api/ensure-ollama`).

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

Seuil τ réglable dans l'interface (défaut `0.95`).

**Important** : les routes `human` / `rejected` **ne produisent pas** d'annotation finale (`related` + `similarity_annotation`) — elles sont **retentées** au prochain Run Model tant qu'elles ne sont pas validées manuellement.

---

## Modèles ML (DeBERTa, SBERT, cross-encoder)

Les poids fine-tunés (~1,5 Go) ne sont **pas** dans Git.

Au premier lancement, `scripts/setup.py` les télécharge depuis la
[Release GitHub v1.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0)
(archive `vldbench-models-v1.tar.gz`).

Sans release accessible :
```bash
export MODELS_DOWNLOAD_URL=https://votre-hebergeur/vldbench-models-v1.tar.gz
```

---

## LLM local

Modèle par défaut : **Qwen2.5-7B** (`qwen2.5:7b-instruct`).

```bash
OLLAMA_LLM_MODEL=mon-modele:tag ./start.sh
```

Variables optionnelles : `.env.example` → `.env` (`OLLAMA_HOST`, `OLLAMA_LLM_MODEL`, `ANNOTATION_DATA_DIR`, etc.).

### Timeouts, retries et mode CPU lent

Configurés dans `cascade/config.json` :

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `inference.timeout` | 600 s | Temps max par appel LLM |
| `inference.llm_retries` | 2 | Retries Ollama |
| `inference.keep_alive` | 30 min | Qwen en RAM entre appels |
| `run_model.no_annotation_timeout` | 360 s | Arrêt si aucune cible annotée |
| `run_model.save_every_n_refs` | 3 | Checkpoint JSON + DB tous les N refs |
| `run_model.cpu_slow.*` | — | Timeouts allongés sur CPU |
| `run_model.estimate.*` | — | Estimation durée (matériel, fractions route, calibration) |

Sur CPU, les cibles LLM peuvent prendre **15–45 s** selon la machine — la barre affiche « running » pendant ce délai.

#### Estimation du temps Run Model

L'estimation s'adapte à **chaque ordinateur** (CPU / GPU CUDA / Apple Silicon) :

| Phase | Affichage UI | Logique |
|-------|--------------|---------|
| **Avant** Run Model (indices Start–End) | `Indices 400–401: estimated time ~21 s` | Durée seule — **pas** de % DeBERTa/LLM (non prédictible) |
| **Après** batch terminé | Résumé : durée réelle + **% DeBERTa / % LLM** mesurés | Basé sur `routing_stats` du batch |

**Calcul** (`POST /api/auto_annotate/estimate`) :

```
temps ≈ démarrage + (cibles restantes × s/cible) + (refs × 0,15 s)
s/cible = Σ (% route × durée/route)
```

| Source | Quand |
|--------|-------|
| Défauts `run_model.estimate` | 1er lancement sur la machine |
| Corpus déjà annoté | % routes tirés du JSON actif |
| Logs session + `instance/estimate_calibration.json` | Après 1+ batch — calibration locale persistante |

Paramètres clés dans `cascade/config.json` :

| Clé | Rôle |
|-----|------|
| `model_load_sec_cpu` / `model_load_sec_gpu` | 1er batch (chargement modèles) |
| `model_load_sec_warm` | Batches suivants (modèles déjà en RAM) |
| `route_fractions` | Répartition théorique DeBERTa / LLM |
| `route_sec_cpu` / `route_sec_gpu` | Durée par route selon matériel |

---

## Utilisation

### 1. Charger un corpus

- **Browse** : ouvre le sélecteur système pour choisir le **dossier de stockage** (où sont enregistrés les nouveaux JSON)
- **Choose JSON file** : ouvre le sélecteur système pour ouvrir un JSON **à son emplacement réel** sur le disque (pas de copie forcée)
- **Storage folder** : chemin affiché ; verrouillable via `ANNOTATION_DATA_DIR` dans `.env`
- **Saved sessions** : registre des fichiers ouverts ou utilisés avec Run Model (`instance/saved_sessions.json`) — **indépendant** du dossier de stockage
- **Active file** : fichier courant en haut du panneau gauche
- **Download JSON** : export du fichier actif

#### Sessions sauvegardées

Chaque fichier JSON **choisi** ou **traité par Run Model** est ajouté au registre. La liste reste visible même après changement de dossier de stockage.

Clic sur **×** : modal avec 3 choix :

| Action | Effet |
|--------|-------|
| **Liste seulement** | Retire l'entrée du registre — le fichier **reste sur l'ordinateur** |
| **Supprimer de l'ordinateur** | Efface définitivement le fichier du disque (+ cache SQLite associé) |
| **Annuler** | Ne rien faire |

### 2. Vérifier la configuration

Badge **All set** requis avant **Run Model**.

### 3. Run Model (annotation automatique)

Bloc d'aide intégré **« How Run Model works »** sous les champs d'index.

1. Indiquer **Start index** et **End index** (positions **0-based** dans le JSON, pas les numéros de batch)
2. Optionnel : ajuster **τ** (défaut 0,95)
3. Cliquer **Run Model** — batch **asynchrone** (HTTP 202)
4. La plage d'indices est **mémorisée par fichier** (persiste après batch / rechargement page)
5. **Run Model reprend automatiquement** à la première référence incomplète de la plage (plus de bouton « Continuer » séparé)
6. **Estimation** sous les indices : durée seule pour la plage choisie (disparaît pendant/après le batch ; résumé avec routing à la fin)
7. **Annuler** / **View logs** depuis l'overlay de progression

#### Comportement skip / re-annotation

| Situation | Comportement |
|-----------|--------------|
| Cible déjà annotée (`related` + score, ou `dismissed`) | Ignorée |
| Cible `human` / `rejected` (pas d'annotation finale) | Retraitée |
| Plage partiellement annotée | Confirmation : ré-annoter (écraser) **ou** compléter les cibles vides seulement |
| Plage 100 % annotée | Confirmation obligatoire pour **ré-annoter** (`force_reannotate: true`) |

#### Annulation

- `POST /api/auto_annotate/cancel` — interruptible (LLM streaming, DeBERTa, chargement modèles)
- Sauvegarde partielle JSON + checkpoint SQLite
- Résumé anglais à la fin (ou partiel si annulé)

### 4. Annotation manuelle

- Navigation **Previous / Next** + liste de références
- **Filtre** : All, Pending, Partial, Complete, Needs validation
- **Next review** : saute aux désaccords `human` / `rejected`
- **Badges cibles** : Pre-filled (auto), Manually annotated, Needs validation, Pending
- **Save & next** / **Dismiss & next**

### 5. Maintenance

- **Clear processing cache (SQLite)** : vide le cache local **sans** supprimer les JSON sur le disque
- **Clear Run Model logs** : vide le log de la session serveur courante (`run_model_session.log`)

Protocole détaillé : `protocole.md`

---

## Où sont stockées les données ?

| Emplacement | Rôle |
|-------------|------|
| Dossier configurable | UI **Browse** ou variable `ANNOTATION_DATA_DIR` (défaut : `uploads/`) |
| JSON sur le disque | **Source de vérité** — peuvent être hors du dossier de stockage |
| `instance/saved_sessions.json` | Registre des sessions (chemins absolus, dernière utilisation) |
| `instance/estimate_calibration.json` | Calibration durées par route (auto, après chaque batch) |
| `{storage}/backups/` | Backup auto avant chaque Run Model |
| `instance/annotations.db` | Cache SQLite (statuts UI) |
| `logs/run_model_session.log` | Log Run Model — **réinitialisé à chaque redémarrage** du serveur |

Champs cascade par cible : `related`, `similarity_annotation`, `cascade_route`, `model_confidence` (+ `llm_pred` / `llm_error` si applicable).

---

## Robustesse

- Verrous fichier JSON (`scripts/annotation_store.py`)
- Sauvegarde incrémentale (checkpoint tous les N refs)
- Skip refs complètes et cibles déjà annotées
- Ré-annotation forcée avec confirmation (`force_reannotate`)
- Backup auto avant batch
- Parsing LLM robuste + retries Ollama
- Annulation interruptible (streaming)
- Progression temps réel (polling `/api/auto_annotate/status` toutes les 2 s)
- Reprise overlay si batch en cours au rechargement page
- Chemins sessions sans altération des noms (`_safe_upload_path` — espaces, parenthèses)
- Estimation durée calibrée par machine (`estimate_calibration.json` + logs session)
- Sélecteurs système cross-platform avec messages d'erreur explicites

---

## API

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Interface web (anglais) |
| GET | `/api/health` | Healthcheck Docker |
| GET | `/api/status` | Checklist configuration |
| POST | `/api/ensure-ollama` | Prépare Ollama + LLM (async) |
| GET | `/api/system-check` | Alias diagnostic |
| POST | `/upload` | Upload JSON (multipart) |
| POST | `/api/upload/pick` | Sélecteur système — ouvre un JSON |
| GET | `/resume` | Dernière session dans le dossier de stockage |
| GET | `/download` | Télécharge le JSON actif |
| POST | `/save_annotation` | Sauvegarde manuelle |
| POST | `/clear_database` | Vide le cache SQLite |
| GET | `/api/sessions` | Liste des sessions sauvegardées |
| POST | `/api/sessions/load` | Charge une session par chemin |
| POST | `/api/sessions/delete` | Retire du registre (`mode: "list"`) ou supprime du disque (`mode: "disk"`) |
| POST | `/api/resync` | Resync SQLite ← JSON |
| GET | `/api/storage` | Dossier de stockage actuel |
| POST | `/api/storage` | Changer le dossier (`{ "path": "/abs/path" }`) |
| POST | `/api/storage/pick` | Sélecteur système — choisir le dossier de stockage |
| GET | `/api/dialogs/capabilities` | Sélecteurs natifs disponibles sur la machine |
| POST | `/auto_annotate` | Lance Run Model (202, `force_reannotate` optionnel) |
| GET | `/api/auto_annotate/status` | Progression du batch |
| POST | `/api/auto_annotate/estimate` | Estimation durée pour une plage d'indices |
| GET | `/api/auto_annotate/resume` | Index de reprise dans une plage |
| POST | `/api/auto_annotate/cancel` | Annulation interruptible |
| GET | `/api/logs/run_model/latest` | Tail du log Run Model |
| POST | `/api/logs/clear` | Vide le log de la session serveur |

---

## Docker

```bash
docker compose up --build
```

Services : `app` (5000) + `ollama` (11434). Volumes : `./models`, `./uploads`, `./logs`.

Voir **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## Dépannage rapide

```bash
python scripts/system_check.py
python scripts/disk_check.py
./start.sh
```

| Problème | Solution |
|----------|----------|
| Interface en français / ancienne version | Redémarrer Flask + **Ctrl+Shift+R** dans le navigateur |
| Browse / Choose JSON ne s'ouvre pas (Linux) | Installer `zenity`, `kdialog` ou `yad` ; vérifier `DISPLAY` / `WAYLAND_DISPLAY` |
| Browse / Choose JSON ne s'ouvre pas (Windows) | Installer PowerShell ou `pwsh` ; Python avec tkinter |
| Sessions sauvegardées vides | Choisir un JSON ou lancer Run Model — la liste se remplit automatiquement |
| Sessions disparaissent après changement de dossier | Corrigé — le registre est indépendant du dossier de stockage |
| Progression 0/0 | Redémarrer Flask (`./start.sh`) |
| Compteur lent sur 0/N | Normal pendant un appel LLM ; barre « running » |
| Run Model indisponible | Compléter la checklist (ML models, Ollama, Qwen) |
| Indices remis à 0–fin après batch | Recharger après mise à jour — plage mémorisée par fichier |
| Session introuvable (parenthèses dans le nom) | Corrigé via `_safe_upload_path` |
| Batch timeout après 1 cible sur CPU | Corrigé (`pairs_evaluated` + timeout CPU 900 s) |
| Annuler sans effet | Attendre ~10 s (streaming LLM) ou recharger la page |
| Batch interrompu | Partiel sauvegardé ; relancer **Run Model** (reprise auto) |
| Estimation trop longue au 1er batch | Normal — se recalibre après le 1er Run Model sur la machine |
| Estimation affichée après batch | Changer Start/End pour la réafficher ; le résumé routing reste sous le batch |
| Logs anciens après redémarrage | Comportement normal — un seul log par session serveur |

---

## Structure du projet

```
annotation-tool-package-2/
├── app.py                    # Flask, routes, SQLAlchemy, registre sessions
├── cascade/                  # Moteur cascade (DeBERTa + LLM)
├── scripts/
│   ├── setup.py
│   ├── run_model_job.py      # Batch async, logs session, estimation
│   ├── annotation_store.py   # Verrous, backups, validation indices
│   ├── native_dialogs.py     # Sélecteurs système (Linux / macOS / Windows)
│   ├── session_discovery.py  # Utilitaire scan disque (optionnel)
│   ├── system_check.py       # Checklist (labels anglais)
│   └── ollama_service.py
├── templates/index.html      # UI anglaise
├── static/app_extras.js      # Sessions, filtres, toasts, sélecteurs
├── uploads/                  # Dossier de stockage par défaut (non versionné)
├── logs/                     # run_model_session.log (non versionné)
├── instance/                 # SQLite + saved_sessions.json (non versionné)
├── models/                   # Poids ML (Release GitHub)
└── start.sh / start.bat / start.ps1
```

---

## Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** — Docker, modèles, dépannage avancé
- **Release modèles** — https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0
- **[AI_annotation](https://github.com/Cespriet/AI_annotation)** — entraînement, évaluation, cascade CLI
