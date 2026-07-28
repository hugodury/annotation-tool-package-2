# ISIALAB Annotation Interface

Application web Flask pour l'annotation VLDBench avec **pipelines d'annotation automatique** :
**Annotate — Qwen only**, **Annotate — Cascade (incl. Qwen)**, **Compare — Qwen ↔ Cascade (incl. Qwen)**.

Interface web **entièrement en anglais**. Fonctionne sur **Windows, macOS et Linux** via un setup unifié (`start.sh` / `start.bat` / `start.ps1`), ou via **Docker**.

| Ressource | Lien |
|-----------|------|
| **Dépôt GitHub** | https://github.com/hugodury/annotation-tool-package-2 |
| **Release modèles base (v1.0.0)** | https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0 |
| **Release Cascade V8 (v8.0.0)** | https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0 |
| **Workspace modèles / cascade CLI** | https://github.com/Cespriet/AI_annotation |

```bash
git clone https://github.com/hugodury/annotation-tool-package-2.git
cd annotation-tool-package-2
```

> **PC neuf / clone GitHub** : une seule commande de démarrage (ci-dessous) suffit.
> Elle crée le venv, installe PyTorch + deps, télécharge les Releases **v1 + v8**, démarre Ollama et tire Qwen.
> **1er lancement** : 15–45 min selon la machine et le réseau. Internet requis une seule fois.
> Les poids ML **ne sont pas dans Git** (~6 Go) — ils viennent des Releases automatiquement.

---

## Démarrage rapide (n'importe quel OS / PC)

| OS | Commande |
|----|----------|
| Mac / Linux | `chmod +x start.sh && ./start.sh` |
| Windows PowerShell | `powershell -ExecutionPolicy Bypass -File .\start.ps1` |
| Windows CMD | `start.bat` |
| Docker | `docker compose up --build` (voir [DEPLOYMENT.md](DEPLOYMENT.md)) |

Puis ouvrir **http://127.0.0.1:5000**

Prérequis manuels avant `./start.sh` : **Python 3.9+** et **Ollama** installés (le script aide à les détecter / tirer le LLM).

> Après modification du code HTML/JS/Python, **redémarrer Flask** si le serveur tournait déjà.

---

## Modes Run Model

| Mode UI | Comportement |
|---------|--------------|
| **Annotate — Qwen only** | Chaque cible → Qwen P3 + post-LLM (pas de Duo / Reranker) |
| **Annotate — Cascade (incl. Qwen)** | Duo MiniLM + DeBERTa Large → Reranker undetermined → **sinon Qwen** (Qwen = dernier étage) |
| **Compare — Qwen ↔ Cascade (incl. Qwen)** | Uniquement ces deux pipelines ; même label → auto ; désaccord → revue |

Détail des règles V8 : [`cascade/CASCADE_RULES.md`](cascade/CASCADE_RULES.md).

---

## Prérequis

| Élément | Détail |
|---------|--------|
| Python | 3.9+ (3.12+ recommandé) — [python.org](https://www.python.org/downloads/) |
| Ollama | [ollama.com](https://ollama.com/) |
| RAM | **≥ 10 Go** recommandés (Qwen 7B) ; 16 Go confortable avec Cascade V8 |
| Disque (1er lancement) | **~20 Go libres** (venv + Release v1 ~1,5 Go + Release v8 ~4 Go + Qwen ~4 Go + marge) |
| Disque (installé) | **~12–14 Go** au total |

Le script de démarrage installe le reste : venv, PyTorch (CPU / CUDA / MPS), modèles ML (v1+v8), pull Qwen via Ollama.

### Cascade V8 + Qwen (MiniLM + DeBERTa Large + Reranker → Qwen)

Installée **automatiquement** au premier démarrage (Release `v8.0.0`), comme les modèles v1.
**Qwen** (Ollama) est le dernier étage de la cascade quand Duo/Reranker ne décident pas.

Pas de symlink machine-dépendante : le dossier `models/cascade_v8/` est un **vrai dossier** recréé sur chaque PC.
Override éventuel : `CASCADE_V8_ROOT` (voir `.env.example`).

Sans Release / sans espace disque, les modes **Cascade** et **Compare** restent
indisponibles ; **Qwen only** fonctionne dès que le LLM Ollama est prêt.

### Storage folder & sélecteurs système

- **Storage folder** : dossier où sont enregistrés les JSON annotés (défaut `uploads/`, ou Bureau via Browse).
- Un fichier ouvert ailleurs (Choose JSON / session hors storage) est **copié** dans ce dossier avant annotation.
- **Browse** / **Choose JSON file** ouvrent le sélecteur natif de l'OS
  (zenity / Finder / PowerShell, avec repli tkinter).

---

## Checklist configuration (interface web)

Au chargement, la checklist vérifie :

- Python, PyTorch, sentence-transformers
- **ML models base (Release v1)** — DeBERTa-base + SBERT + cross-encoder (SBERT pour similarité)
- **Cascade V8 models (Release v8)** — MiniLM + DeBERTa Large + Reranker
- Ollama installé / running + LLM `qwen2.5:7b-instruct` (étage final Cascade / Compare / Qwen only)

**Run Model** n'est disponible que si tous les items **requis** sont ✓.

---

## Routage cascade

### Cascade (`v8_qwen`) — inclut Qwen

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

### Compare (`compare`)

Exécute **Qwen only** et **Cascade (incl. Qwen)** :

- même label → `compare_agree` (auto-annotation)
- labels différents → `compare_disagree` (revue + détail side-by-side)

Les routes `human` / `rejected` / `compare_disagree` **ne finalisent pas** l'annotation ;
elles sont retentées au prochain Run Model.

---

## Modèles ML — Git exclus, Releases multi-OS

Les poids **ne sont jamais dans Git** (`.gitignore` → `/models/`).  
`./start.sh` / `start.ps1` / `start.bat` appellent `scripts/download_models.py`, qui enchaîne **v1 puis v8** sur Windows, macOS et Linux.

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

Détail multi-OS / dépannage : **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## LLM local

Modèle par défaut : **Qwen2.5-7B** (`qwen2.5:7b-instruct`).

```bash
OLLAMA_LLM_MODEL=mon-modele:tag ./start.sh
```

Variables optionnelles : copier `.env.example` → `.env`
(`OLLAMA_HOST`, `OLLAMA_LLM_MODEL`, `ANNOTATION_DATA_DIR`, `CASCADE_V8_ROOT`, …).

Timeouts / retries / estimation : `cascade/config.json`
(`inference.*`, `run_model.*`, `cascade_v8.*`).

---

## Utilisation

1. Choisir le **Storage folder** (Browse) — les JSON annotés y seront écrits
2. **Charger un corpus** (Choose JSON / Saved sessions) — copie dans le storage si besoin
3. Checklist **All set**
4. **Run Model** : indices 0-based + un des 3 modes
5. Annotation manuelle : filtres, Next review, Save / Dismiss

Protocole P3 : `cascade/protocol.md` · Few-shot : `cascade/few_shot.json`

---

## Données

| Emplacement | Rôle |
|-------------|------|
| Storage folder (`ANNOTATION_DATA_DIR` ou réglage UI / défaut `uploads/`) | JSON annotés |
| `instance/storage_settings.json` | Dossier storage choisi |
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
| POST | `/auto_annotate` | Lance Run Model (`cascade_mode`: `qwen_only` \| `v8_qwen` \| `compare`) |
| GET | `/api/auto_annotate/status` | Progression |
| POST | `/api/auto_annotate/estimate` | Estimation durée |
| POST | `/api/auto_annotate/cancel` | Annulation |
| GET | `/api/system-check` | Checklist configuration |
| GET | `/api/storage` | Dossier storage actuel |

---

## Docker

```bash
docker compose up --build
```

Voir **[DEPLOYMENT.md](DEPLOYMENT.md)** (service app + Ollama).

---

## Structure

```
annotation-tool-package-2/
├── app.py
├── cascade/
│   ├── core.py              # modes qwen_only / v8_qwen / compare
│   ├── v8_predictor.py      # Duo Zero Faute + Reranker
│   ├── CASCADE_RULES.md
│   ├── protocol.md / few_shot.json / config.json
│   └── post_llm_regles.txt
├── scripts/
│   ├── setup.py             # install cross-OS
│   ├── download_models.py   # v1 puis v8
│   ├── download_models_v8.py
│   └── run_model_job.py
├── templates/index.html
├── models/                  # créé au 1er start (Releases) — gitignored
└── start.sh / start.bat / start.ps1
```

---

## Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** — install multi-OS, Releases v1 + **v8**, Docker, dépannage
- **[cascade/CASCADE_RULES.md](cascade/CASCADE_RULES.md)** — règles Duo / Reranker / Qwen
- **Release modèles v1** — https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0
- **Release Cascade V8** — https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0
- **[AI_annotation](https://github.com/Cespriet/AI_annotation)** — entraînement, évaluation, sources V8
