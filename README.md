# ISIALAB Annotation Interface

Application web Flask pour l'annotation VLDBench avec **pipelines d'annotation automatique** :
**Qwen only**, **DeBERTa-base + Qwen**, **Cascade V8 + Qwen** (MiniLM + DeBERTa Large + Reranker → Qwen), et **Compare** (Qwen only ↔ Cascade V8).

Interface web **entièrement en anglais**. Fonctionne sur **Windows, macOS et Linux** via un setup unifié (`start.sh` / `start.bat` / `start.ps1`), ou via **Docker**.

| Ressource | Lien |
|-----------|------|
| **Dépôt GitHub** | https://github.com/hugodury/annotation-tool-package-2 |
| **Release modèles ML (v1.0.0)** | https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0 |
| **Workspace modèles / cascade CLI** | https://github.com/Cespriet/AI_annotation |

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

> Après modification du code HTML/JS/Python, **redémarrer Flask** si le serveur tournait déjà.

---

## Modes Run Model

| Mode UI | Comportement |
|---------|--------------|
| **Qwen only** | Chaque cible → Qwen P3 + post-LLM |
| **DeBERTa + Qwen** | DeBERTa-base (≥ τ auto) ; sinon Qwen sur supporting/undetermined |
| **Cascade V8 + Qwen** | Duo MiniLM+DeBERTa Large → Reranker undetermined → sinon Qwen P3 |
| **Compare Qwen ↔ Cascade V8** | Lance les deux pipelines ; accord → auto ; désaccord → revue |

Détail des règles V8 : [`cascade/CASCADE_RULES.md`](cascade/CASCADE_RULES.md).

---

## Prérequis

| Élément | Détail |
|---------|--------|
| Python | 3.9+ (3.12+ recommandé) — [python.org](https://www.python.org/downloads/) |
| Ollama | [ollama.com](https://ollama.com/) |
| RAM | 10 Go recommandés (Qwen 7B) ; + marge pour Cascade V8 (~4 Go de poids locaux) |
| Disque (1er lancement) | **~13 Go libres** (venv, modèles ML ~1,5 Go, LLM ~4 Go, marge) |
| Disque (une fois installé) | **~8 Go** au total (+ ~4 Go si Cascade V8 installée à côté) |

Le script de démarrage installe le reste : venv, PyTorch (CPU / CUDA / MPS), modèles ML, pull Qwen via Ollama.

### Cascade V8 (MiniLM + DeBERTa Large + Reranker)

Installée **automatiquement** au premier démarrage (Release `v8.0.0`), comme les modèles v1.

Pas de symlink machine-dépendante : le dossier `models/cascade_v8/` est créé sur chaque PC.
Override éventuel : `CASCADE_V8_ROOT` (voir `.env.example`).

Sans Release / sans espace disque, les modes **Cascade V8 + Qwen** et **Compare** restent
indisponibles ; **Qwen only** fonctionne dès que le LLM Ollama est prêt.

### Sélecteurs système (Browse / Choose JSON)

Les boutons **Browse** et **Choose JSON file** ouvrent le **sélecteur natif** de l'OS
(zenity / Finder / PowerShell, avec repli tkinter).

---

## Checklist configuration (interface web)

Au chargement, une checklist vérifie Python, PyTorch, sentence-transformers, modèles ML,
Ollama et le LLM `qwen2.5:7b-instruct`. **Run Model** n'est disponible que si tout est ✓.

---

## Routage cascade

### Cascade V8 + Qwen (`v8_qwen`)

```text
Pair (T_ref, T_n)
  │
  ├─ Duo MiniLM (0.4) + DeBERTa Large (0.6)
  │     Rule 5 disagree → reject
  │     Rule 1 P(against)>0.35 → AUTO against          (route v8_duo)
  │     Rule 2 P(undet)>0.10 → reject
  │     Rule 3 max P≥0.98 → AUTO                       (route v8_duo)
  │     Rule 4 sinon → reject
  │
  ├─ Reranker undetermined (si reject)
  │     P≥0.70 → AUTO undetermined                     (route v8_reranker)
  │
  └─ sinon → Qwen P3 + post-LLM                        (route v8_qwen)
```

DeBERTa Large tourne en **CPU** (NaNs observés sur GPU/MPS).

### DeBERTa-base + Qwen (`deberta_qwen`)

| Situation | Route | Comportement |
|-----------|-------|--------------|
| DeBERTa confiant (≥ τ, défaut 95 %) | `deberta_auto` | Annotation automatique |
| Classe `against` / `not_related` ambiguë | `deberta_ambiguous` | DeBERTa seul |
| Classe `supporting` / `undetermined` ambiguë | `consensus` / `human` | LLM Qwen (P3) |
| Désaccord fort (≥ 85 %) | `rejected` | Revue humaine |
| Timeout / erreur LLM | `human` | Revue humaine |

Seuil τ réglable dans l'UI — **uniquement** pour ce mode.

### Compare (`compare`)

Exécute **Qwen only** et **Cascade V8 + Qwen** en parallèle :

- même label → `compare_agree` (auto-annotation)
- labels différents → `compare_disagree` (revue + détail)

Les routes `human` / `rejected` / `compare_disagree` **ne finalisent pas** l'annotation ;
elles sont retentées au prochain Run Model.

---

## Modèles ML — Git exclus, Releases multi-OS

Les poids **ne sont jamais dans Git** (`.gitignore` → `/models/`).  
`./start.sh` / `start.ps1` / `start.bat` les téléchargent sur **Windows, macOS et Linux**.

| Release | Tag | Contenu | Script |
|---------|-----|---------|--------|
| Base | [v1.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0) | DeBERTa-base, SBERT, cross-encoder (~1,5 Go) | `download_models.py` |
| **Cascade V8** | [v8.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0) | MiniLM + DeBERTa Large + Reranker (~4 Go) | `download_models_v8.py` |

Après install :

```text
models/
├── fine_tuned_*                 # v1
└── cascade_v8/                  # v8 — dossier réel (pas de symlink absolu)
    ├── config.json
    └── models/{minilm_full_v7,deberta_large_v8.1,reranker_undetermined_v8}/
```

Le Reranker dépasse la limite GitHub **2 Go/fichier** : il est découpé en `.part00`/`.part01`
puis réassemblé automatiquement. Checksums : `models.manifest.json`, `models-v8.manifest.json`.

Miroir custom : `MODELS_DOWNLOAD_URL` et `CASCADE_V8_DOWNLOAD_URL_*` (voir `.env.example`).

**Mainteneur** — republier les poids :

```bash
python scripts/package_models.py          # v1
python scripts/package_models_v8.py       # v8 assets → dist/cascade-v8/
python scripts/publish_models_v8.py       # gh release v8.0.0
```

Détail multi-OS : **[DEPLOYMENT.md](DEPLOYMENT.md)**.

Disque 1er lancement : **~20 Go libres** recommandés (venv + v1 + v8 + Qwen).

---

## LLM local

Modèle par défaut : **Qwen2.5-7B** (`qwen2.5:7b-instruct`).

```bash
OLLAMA_LLM_MODEL=mon-modele:tag ./start.sh
```

Variables optionnelles : `.env.example` → `.env`
(`OLLAMA_HOST`, `OLLAMA_LLM_MODEL`, `ANNOTATION_DATA_DIR`, `CASCADE_V8_ROOT`, …).

Timeouts / retries / estimation : `cascade/config.json`
(`inference.*`, `run_model.*`, `cascade_v8.*`).

---

## Utilisation

1. **Charger un corpus** (Browse / Choose JSON / Saved sessions)
2. Checklist **All set**
3. **Run Model** : indices 0-based + mode pipeline (+ τ si DeBERTa-base)
4. Annotation manuelle : filtres, Next review, Save / Dismiss

Protocole P3 : `cascade/protocol.md` · Few-shot : `cascade/few_shot.json`

---

## Données

| Emplacement | Rôle |
|-------------|------|
| `ANNOTATION_DATA_DIR` / `uploads/` | Stockage JSON |
| `instance/saved_sessions.json` | Registre sessions |
| `instance/estimate_calibration.json` | Calibration estimation |
| `instance/annotations.db` | Cache SQLite |
| `logs/run_model_session.log` | Log Run Model |

Champs par cible : `related`, `similarity_annotation`, `cascade_route`,
`model_confidence`, éventuellement `cascade_v8` / `pipeline_compare`.

---

## API (extrait)

| Méthode | Route | Description |
|---------|-------|-------------|
| POST | `/auto_annotate` | Lance Run Model (`cascade_mode`, `force_reannotate`) |
| GET | `/api/auto_annotate/status` | Progression |
| POST | `/api/auto_annotate/estimate` | Estimation durée |
| POST | `/api/auto_annotate/cancel` | Annulation |
| GET | `/api/status` | Checklist |

---

## Docker

```bash
docker compose up --build
```

Voir **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## Structure

```
annotation-tool-package-2/
├── app.py
├── cascade/
│   ├── core.py              # modes qwen_only / deberta_qwen / v8_qwen / compare
│   ├── v8_predictor.py      # Duo Zero Faute + Reranker
│   ├── CASCADE_RULES.md
│   ├── protocol.md / few_shot.json / config.json
│   └── post_llm_regles.txt
├── scripts/run_model_job.py
├── templates/index.html
├── models/                  # Release ML + symlink cascade_v8
└── start.sh / start.bat / start.ps1
```

---

## Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** — Docker, modèles, dépannage
- **[cascade/CASCADE_RULES.md](cascade/CASCADE_RULES.md)** — règles Duo / Reranker / Qwen
- **Release modèles** — https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0
- **[AI_annotation](https://github.com/Cespriet/AI_annotation)** — entraînement, évaluation, package Cascade V8
