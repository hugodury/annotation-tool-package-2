# ISIALAB Annotation Interface

Application web Flask pour l'annotation VLDBench avec **pipelines d'annotation automatique** :

| Mode UI | Code API | Comportement |
|---------|----------|--------------|
| **Annotate — Qwen only** | `qwen_only` | Chaque cible → Qwen P3 + post-LLM |
| **Annotate — Cascade (incl. Qwen)** | `v8_qwen` | Duo MiniLM + DeBERTa Large → Reranker → **sinon Qwen** |
| **Compare — Qwen ↔ Cascade (incl. Qwen)** | `compare` | Les deux pipelines ; accord → auto ; désaccord → revue |

Interface web **entièrement en anglais**. Compatible **Windows, macOS et Linux** (`start.sh` / `start.bat` / `start.ps1`) ou **Docker**.

| Ressource | Lien |
|-----------|------|
| **Dépôt GitHub** | https://github.com/hugodury/annotation-tool-package-2 |
| **Release modèles base (v1.0.0)** | https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0 |
| **Release Cascade V8 (v8.0.0)** | https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0 |
| **Workspace modèles / cascade CLI** | https://github.com/Cespriet/AI_annotation |
| **Déploiement multi-OS** | [DEPLOYMENT.md](DEPLOYMENT.md) |

```bash
git clone https://github.com/hugodury/annotation-tool-package-2.git
cd annotation-tool-package-2
```

> **PC neuf / clone GitHub** : une seule commande de démarrage (ci-dessous).
> Elle crée le venv, installe PyTorch + deps, télécharge les Releases **v1 + v8**, démarre Ollama et tire Qwen.
> **1er lancement** : 15–45 min (réseau / machine). Internet requis **une seule fois**.
> Les poids ML **ne sont pas dans Git** (~6 Go) — Releases GitHub automatiques.
> Après install, **~5–6 Go libres** sur le disque suffisent : le message « ~20 Go » ne s’applique qu’au **premier** install.

---

## Démarrage rapide (n'importe quel OS / PC)

| OS | Commande |
|----|----------|
| Mac / Linux | `chmod +x start.sh && ./start.sh` |
| Windows PowerShell | `powershell -ExecutionPolicy Bypass -File .\start.ps1` |
| Windows CMD | `start.bat` |
| Docker | `docker compose up --build` → [DEPLOYMENT.md](DEPLOYMENT.md) |

Puis ouvrir **http://127.0.0.1:5000**

**Prérequis manuels** : Python **3.9+** et [Ollama](https://ollama.com/) (le script détecte / tire le LLM).

> Après modification HTML / JS / Python : **redémarrer Flask** si le serveur tournait déjà.

---

## Prérequis

| Élément | Détail |
|---------|--------|
| Python | 3.9+ (3.12+ recommandé) — [python.org](https://www.python.org/downloads/) |
| Ollama | [ollama.com](https://ollama.com/) |
| RAM | **≥ 10 Go** recommandés (Qwen 7B) ; 16 Go confortable avec Cascade |
| Disque (1er lancement) | **~20 Go libres** (venv + v1 ~1,5 Go + v8 ~4 Go + Qwen ~4 Go + marge) |
| Disque (déjà installé) | **~12–14 Go** utilisés ; l’espace libre restant peut être faible (normal) |
| GPU | Optionnel (accélération). Cascade DeBERTa Large tourne en **CPU** (stabilité) |

Le script installe : venv, PyTorch (CPU / CUDA / MPS), modèles ML (v1+v8), pull Qwen.

---

## Checklist configuration (UI)

Au chargement (`/api/system-check`) :

| Item | Requis | Rôle |
|------|--------|------|
| System / Python / PyTorch / sentence-transformers ≥ 5.5 | oui | Runtime |
| **ML models base (Release v1)** | oui | SBERT (similarité) + package base |
| **Cascade V8 models (Release v8)** | oui | Duo + Reranker pour Cascade / Compare |
| Ollama installé + running | oui | Serveur LLM |
| **LLM qwen2.5:7b-instruct** | oui | Qwen only + dernier étage Cascade + Compare |
| RAM ≥ 10 Go | non | Recommandé |
| Disk space | non | Info ; « install complete » si déjà installé |
| GPU / Estimated performance | non | Info (CPU → Slow) |

**Run Model** n’est actif que si tous les items **requis** sont ✓ (**All set**).

---

## Modes Run Model (détail)

### 1. Annotate — Qwen only (`qwen_only`)

Chaque paire (référence, cible) passe par **Qwen P3 + post-LLM**.
Pas de Duo / Reranker.

### 2. Annotate — Cascade incl. Qwen (`v8_qwen`)

```text
Pair (T_ref, T_n)
  │
  ├─ Duo MiniLM (0.4) + DeBERTa Large (0.6)   « Zero Faute »
  │     Rule 5 disagree → reject
  │     Rule 1 P(against)>0.35 → AUTO against     (route v8_duo)
  │     Rule 2 P(undet)>0.10 → reject
  │     Rule 3 max P≥0.98 → AUTO                  (route v8_duo)
  │     Rule 4 sinon → reject
  │
  ├─ Reranker undetermined (si reject)
  │     P≥0.70 → AUTO undetermined                (route v8_reranker)
  │
  └─ sinon → Qwen P3 + post-LLM                   (route v8_qwen)
```

**Qwen fait partie de la cascade** (dernier étage). Seuils dans `models/cascade_v8/config.json`.
Règles écrites : [`cascade/CASCADE_RULES.md`](cascade/CASCADE_RULES.md).

### 3. Compare — Qwen ↔ Cascade (`compare`)

Lance **uniquement** ces deux pipelines (pas d’ancien DeBERTa-base) :

| Résultat | Route | Action |
|----------|-------|--------|
| Même label | `compare_agree` | Auto-annotation |
| Labels différents | `compare_disagree` | Revue + détail side-by-side |

Routes **non finales** (retentées au prochain run) : `human`, `rejected`, `compare_disagree`.

---

## Estimation de durée

| Situation | Affichage |
|-----------|-----------|
| **Premier run** d’un mode (Qwen only *ou* Cascade) | Pas d’estimation théorique — message *finish a first Run Model with this mode…* |
| Après un run **Qwen only** | Estimation pour **Qwen only** (échantillon calibré) |
| Après un run **Cascade** | Estimation pour **Cascade** (calibrage **séparé**) |
| **Compare** | Pas d’estimation unique (2 pipelines) |

Calibration persistée dans `instance/estimate_calibration.json` (par mode, fichier, machine).
Survit aux redémarrages Flask.

---

## Storage folder & fichiers

- **Storage folder** (Browse) : dossier des JSON annotés (défaut `uploads/`, ou Bureau, etc.).
- Fichier ouvert ailleurs (**Choose JSON** / session hors storage) → **copié** dans le storage avant annotation.
- **Browse** / **Choose JSON** : sélecteur natif (zenity / Finder / PowerShell + repli tkinter).
- Verrouillage possible : `ANNOTATION_DATA_DIR` dans `.env`.

---

## Utilisation (parcours type)

1. Checklist **All set**
2. Choisir le **Storage folder**
3. Charger un corpus (**Choose JSON** / Saved sessions)
4. **Run Model** : indices **0-based** (positions JSON, pas numéros de batch) + mode
5. Reprise automatique au premier référence incomplet ; cibles déjà faites **sautées** (sauf overwrite)
6. Si la plage est déjà annotée → modal de confirmation (reprendre / tout réannoter)
7. Revue manuelle : filtres, **Next review**, Save & next / Dismiss

Protocole LLM : `cascade/protocol.md` · Few-shot : `cascade/few_shot.json` · Post-LLM : `cascade/post_llm_regles.txt`

---

## Modèles ML — Releases multi-OS (hors Git)

`.gitignore` → `/models/`.  
`./start.sh` appelle `scripts/download_models.py` → **v1 puis v8**.

| Release | Tag | Contenu | Script |
|---------|-----|---------|--------|
| Base | [v1.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0) | DeBERTa-base, SBERT, cross-encoder (~1,5 Go) | `download_models.py` |
| **Cascade V8** | [v8.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0) | MiniLM + DeBERTa Large + Reranker (~4 Go) | `download_models_v8.py` |

```text
models/
├── fine_tuned_*                 # v1
└── cascade_v8/                  # v8 — vrai dossier (pas de symlink absolu)
    ├── config.json
    └── models/{minilm_full_v7,deberta_large_v8.1,reranker_undetermined_v8}/
```

Reranker > 2 Go GitHub → `.part00` / `.part01` réassemblés auto.  
Checksums : `models.manifest.json`, `models-v8.manifest.json`.  
Miroirs : `MODELS_DOWNLOAD_URL`, `CASCADE_V8_DOWNLOAD_URL_*` (`.env.example`).

**Mainteneur** :

```bash
python scripts/package_models.py
python scripts/package_models_v8.py
python scripts/publish_models_v8.py
```

Sans Release / sans disque : **Cascade** et **Compare** indisponibles ; **Qwen only** OK dès qu’Ollama + Qwen sont prêts.

---

## LLM local (Ollama)

Défaut : **`qwen2.5:7b-instruct`**.

```bash
OLLAMA_LLM_MODEL=mon-modele:tag ./start.sh
```

Copier `.env.example` → `.env` :

| Variable | Rôle |
|----------|------|
| `OLLAMA_HOST` | URL Ollama (défaut `http://127.0.0.1:11434`) |
| `OLLAMA_LLM_MODEL` | Tag modèle |
| `ANNOTATION_DATA_DIR` | Force le storage folder (verrouille Browse) |
| `CASCADE_V8_ROOT` | Override chemin Cascade V8 |
| `CASCADE_V8_DOWNLOAD_URL_*` | Miroirs assets v8 |
| `MODELS_DOWNLOAD_URL` | Miroir archive v1 |
| `PYTORCH_INDEX_URL` | Index pip PyTorch |
| `FLASK_HOST` / `FLASK_PORT` | Bind serveur |

Timeouts / retries / estimation théorique interne : `cascade/config.json`
(`inference.*`, `run_model.*`, `cascade_v8.*`).

---

## Données & chemins

| Emplacement | Rôle |
|-------------|------|
| Storage folder (UI / `ANNOTATION_DATA_DIR` / défaut `uploads/`) | JSON annotés + `backups/` |
| `instance/storage_settings.json` | Dossier storage choisi |
| `instance/storage_history.json` | Historique dossiers |
| `instance/saved_sessions.json` | Registre sessions |
| `instance/estimate_calibration.json` | Calibration temps **par mode** |
| `instance/annotations.db` | Cache SQLite (Clear processing cache ne touche pas les JSON) |
| `logs/run_model_session.log` | Log Run Model de la session serveur |

**Champs typiques par cible** : `related`, `similarity_annotation`, `cascade_route`,
`model_confidence`, `llm_pred`, éventuellement `cascade_v8`, `pipeline_compare`
(avec côté Qwen + côté Cascade pour Compare).

---

## API (extrait)

| Méthode | Route | Description |
|---------|-------|-------------|
| POST | `/auto_annotate` | Lance Run Model (`cascade_mode`, `force_reannotate`, indices) |
| GET | `/api/auto_annotate/status` | Progression / résumé |
| POST | `/api/auto_annotate/estimate` | Estimation (seulement si mode déjà calibré) |
| POST | `/api/auto_annotate/cancel` | Annulation |
| GET | `/api/system-check` | Checklist |
| GET/POST | `/api/storage` | Lire / fixer le storage |
| POST | `/api/storage/pick` | Browse natif |
| POST | `/api/upload/pick` | Choose JSON (copie vers storage) |
| GET | `/api/sessions` | Sessions sauvegardées |
| POST | `/api/sessions/load` | Recharger (copie vers storage si besoin) |
| GET | `/api/dialogs/capabilities` | Sélecteurs OS disponibles |

---

## Docker

```bash
docker compose up --build
```

App + Ollama : détails dans **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## Structure du dépôt

```
annotation-tool-package-2/
├── app.py                       # Flask + storage + API
├── cascade/
│   ├── core.py                  # qwen_only / v8_qwen / compare
│   ├── v8_predictor.py          # Duo Zero Faute + Reranker
│   ├── CASCADE_RULES.md
│   ├── protocol.md / few_shot.json / config.json
│   └── post_llm_regles.txt
├── scripts/
│   ├── setup.py                 # install cross-OS
│   ├── download_models.py       # v1 puis v8
│   ├── download_models_v8.py
│   ├── run_model_job.py         # batch + calibration estimation
│   ├── system_check.py          # checklist
│   └── native_dialogs.py        # sélecteurs OS
├── templates/index.html         # UI (anglais)
├── static/app_extras.js
├── models/                      # créé au 1er start — gitignored
├── instance/                    # settings locaux — souvent gitignored
├── start.sh / start.bat / start.ps1
├── docker-compose.yml / Dockerfile
├── README.md
└── DEPLOYMENT.md
```

---

## Dépannage rapide

| Problème | Piste |
|----------|-------|
| Run Model grisé | Checklist : modèles v1/v8, Ollama, Qwen |
| Cascade / Compare KO, Qwen OK | `python scripts/download_models_v8.py` ou espace disque |
| Pas d’estimation de temps | Normal au 1er run du mode — lancer une fois puis réessayer |
| Fichiers toujours dans un vieux `uploads/` | Vérifier Storage folder ; rouvrir le JSON (copie vers storage) |
| Sélecteur de dossier KO | Installer zenity/kdialog (Linux) ou laisser tkinter ; voir README dialogs |
| Port 5000 occupé | Arrêter l’ancien `python app.py` / `fuser -k 5000/tcp` |
| Clone sans modèles | Normal — lancer `./start.sh` (pas de poids dans Git) |

Plus de détail : **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## Documentation liée

- **[DEPLOYMENT.md](DEPLOYMENT.md)** — install multi-OS, Releases v1+v8, Docker, dépannage
- **[cascade/CASCADE_RULES.md](cascade/CASCADE_RULES.md)** — règles Duo / Reranker / Qwen
- **Release v1** — https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0
- **Release Cascade V8** — https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0
- **[AI_annotation](https://github.com/Cespriet/AI_annotation)** — entraînement, évaluation, sources V8
