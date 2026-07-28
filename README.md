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
| **Release Cascade V8 (v8.0.0)** | https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0 |
| **Workspace modèles / cascade CLI** | https://github.com/Cespriet/AI_annotation |
| **Déploiement multi-OS** | [DEPLOYMENT.md](DEPLOYMENT.md) |

```bash
git clone https://github.com/hugodury/annotation-tool-package-2.git
cd annotation-tool-package-2
```

> **PC neuf / clone GitHub** : une seule commande de démarrage (ci-dessous).
> Elle crée le venv, installe PyTorch + deps, télécharge **Cascade V8**, démarre Ollama et tire Qwen.
> **1er lancement** : 15–45 min (réseau / machine). Internet requis **une seule fois**.
> Les poids ML **ne sont pas dans Git** — Release GitHub `v8.0.0` automatique.
> L’ancienne Release **v1** (DeBERTa-base / SBERT) n’est **plus utilisée** par Run Model.

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
| RAM | **≥ 16 Go** recommandés pour **Cascade + Qwen** (non bloquant) ; ≥ **10 Go** minimum pour Qwen only |
| Disque (1er lancement) | **~15 Go libres** (venv + Cascade V8 ~4–7 Go + Qwen ~4 Go + marge) |
| Disque (déjà installé) | espace libre restant peut être faible (normal) |
| GPU | Optionnel. DeBERTa Large (Cascade) tourne en **CPU** (stabilité) |

Le script installe : venv, PyTorch (CPU / CUDA / MPS), modèles Cascade V8, pull Qwen.

---

## Checklist configuration (UI)

Au chargement (`/api/system-check`) :

| Item | Requis | Rôle |
|------|--------|------|
| System / Python / PyTorch / sentence-transformers ≥ 5.5 | oui | Runtime |
| **Cascade V8 models (Release v8)** | oui | Duo + Reranker pour Cascade / Compare |
| Ollama installé + running | oui | Serveur LLM |
| **LLM qwen2.5:7b-instruct** | oui | Qwen only + dernier étage Cascade + Compare |
| RAM ≥ 16 Go (Cascade) | **non** | Recommandé seulement — n’empêche pas Run Model |
| Disk space | non | Info ; « install complete » si déjà installé |
| GPU / Estimated performance | non | Info (CPU → Slow) |

**Run Model** est actif dès que tous les items **requis** sont ✓ (**All set**).  
La RAM ○ / ✓ est purement informative.

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
Règles : [`cascade/CASCADE_RULES.md`](cascade/CASCADE_RULES.md).

**Qui a annoté ?** Chaque cible affiche un badge clair :

| Badge UI | Route / champ | Décideur |
|----------|---------------|----------|
| **By: Duo** | `v8_duo` / `annotated_by: duo` | MiniLM + DeBERTa Large |
| **By: Reranker** | `v8_reranker` / `annotated_by: reranker` | Reranker undetermined |
| **By: Qwen (Cascade)** | `v8_qwen` / `annotated_by: qwen` | Fallback LLM |

Détail stage / règle / MiniLM / DeBERTa dans l’UI + JSON (`cascade_v8`, `annotated_by_label`).

### 3. Compare — Qwen ↔ Cascade (`compare`)

Lance **uniquement** ces deux pipelines :

| Résultat | Route | Action |
|----------|-------|--------|
| Même label | `compare_agree` | Auto-annotation |
| Labels différents | `compare_disagree` | Revue + détail side-by-side |

Routes **non finales** (retentées au prochain run) : `human`, `rejected`, `compare_disagree`.

---

## Estimation de durée

| Situation | Affichage |
|-----------|-----------|
| **Premier run** d’un mode | Pas d’estimation théorique — *finish a first Run Model with this mode…* |
| Après un run **Qwen only** | Estimation pour **Qwen only** |
| Après un run **Cascade** | Estimation pour **Cascade** (calibrage **séparé**) |
| **Compare** | Pas d’estimation unique |

Calibration : `instance/estimate_calibration.json` (par mode). Survit aux redémarrages Flask.

---

## Storage folder & fichiers

- **Storage folder** (Browse) : dossier des JSON annotés (défaut `uploads/`, ou Bureau, etc.).
- Fichier ouvert ailleurs → **copié** dans le storage avant annotation.
- Sélecteurs natifs OS (zenity / Finder / PowerShell + tkinter).
- Verrouillage : `ANNOTATION_DATA_DIR` dans `.env`.

---

## Utilisation (parcours type)

1. Checklist **All set**
2. Choisir le **Storage folder**
3. Charger un corpus (**Choose JSON** / Saved sessions)
4. **Run Model** : indices **0-based** + mode
5. Reprise au premier référence incomplet ; cibles faites sautées (sauf overwrite)
6. Plage déjà annotée → modal (reprendre / tout réannoter)
7. Revue : filtres, **Next review**, Save & next / Dismiss — badges **By: Duo / Reranker / Qwen**

Protocole : `cascade/protocol.md` · Few-shot : `cascade/few_shot.json` · Post-LLM : `cascade/post_llm_regles.txt`

---

## Modèles ML — Releases multi-OS (hors Git)

`.gitignore` → `/models/`.  
Run Model actuel nécessite **uniquement Cascade V8** (+ Qwen via Ollama).

| Release | Tag | Contenu | Requis Run Model ? |
|---------|-----|---------|-------------------|
| **Cascade V8** | [v8.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0) | MiniLM + DeBERTa Large + Reranker (~4 Go) | **Oui** |
| Base v1 (legacy) | [v1.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0) | DeBERTa-base, SBERT, cross-encoder | **Non** (plus dans la checklist UI) |

```text
models/
└── cascade_v8/
    ├── config.json
    └── models/{minilm_full_v7,deberta_large_v8.1,reranker_undetermined_v8}/
```

Reranker > 2 Go GitHub → `.part00` / `.part01` réassemblés auto.  
Checksums : `models-v8.manifest.json`. Miroirs : `CASCADE_V8_DOWNLOAD_URL_*` (`.env.example`).

```bash
python scripts/download_models_v8.py   # utilisateur
python scripts/package_models_v8.py    # mainteneur
python scripts/publish_models_v8.py
```

Sans V8 : **Cascade** / **Compare** indisponibles ; **Qwen only** OK dès qu’Ollama + Qwen sont prêts.

---

## LLM local (Ollama)

Défaut : **`qwen2.5:7b-instruct`**.

```bash
OLLAMA_LLM_MODEL=mon-modele:tag ./start.sh
```

| Variable `.env` | Rôle |
|-----------------|------|
| `OLLAMA_HOST` | URL Ollama |
| `OLLAMA_LLM_MODEL` | Tag modèle |
| `ANNOTATION_DATA_DIR` | Force le storage folder |
| `CASCADE_V8_ROOT` | Override chemin Cascade V8 |
| `CASCADE_V8_DOWNLOAD_URL_*` | Miroirs assets v8 |
| `PYTORCH_INDEX_URL` | Index pip PyTorch |
| `FLASK_HOST` / `FLASK_PORT` | Bind serveur |

Config : `cascade/config.json` (`inference.*`, `run_model.*`, `cascade_v8.*`).

---

## Données & chemins

| Emplacement | Rôle |
|-------------|------|
| Storage folder | JSON annotés + `backups/` |
| `instance/storage_settings.json` | Dossier storage choisi |
| `instance/saved_sessions.json` | Registre sessions |
| `instance/estimate_calibration.json` | Calibration temps **par mode** |
| `instance/annotations.db` | Cache SQLite |
| `logs/run_model_session.log` | Log Run Model |

**Champs par cible** : `related`, `similarity_annotation`, `cascade_route`,
`model_confidence`, `annotated_by`, `annotated_by_label`, `cascade_v8`,
éventuellement `pipeline_compare` (Compare).

---

## API (extrait)

| Méthode | Route | Description |
|---------|-------|-------------|
| POST | `/auto_annotate` | Run Model (`qwen_only` \| `v8_qwen` \| `compare`) |
| GET | `/api/auto_annotate/status` | Progression |
| POST | `/api/auto_annotate/estimate` | Estimation (si mode déjà calibré) |
| POST | `/api/auto_annotate/cancel` | Annulation |
| GET | `/api/system-check` | Checklist |
| GET/POST | `/api/storage` | Storage folder |
| POST | `/api/storage/pick` | Browse natif |
| POST | `/api/upload/pick` | Choose JSON |
| GET | `/api/sessions` | Sessions |
| POST | `/api/sessions/load` | Recharger session |

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
├── cascade/          # core, v8_predictor, protocol, CASCADE_RULES.md
├── scripts/          # setup, download_models_v8, run_model_job, system_check
├── templates/        # UI anglais
├── models/           # gitignored — créé au 1er start (Release v8)
├── start.sh / start.bat / start.ps1
├── README.md / DEPLOYMENT.md
└── docker-compose.yml
```

---

## Dépannage rapide

| Problème | Piste |
|----------|-------|
| Run Model grisé | Checklist : Cascade V8, Ollama, Qwen |
| Cascade / Compare KO | `python scripts/download_models_v8.py` |
| RAM ○ à 15 Go | Normal — reco Cascade ≥16 Go, **non bloquant** |
| Pas d’estimation | 1er run du mode — calibrer puis réessayer |
| Fichiers hors storage | Rouvrir le JSON (copie auto) |
| Port 5000 occupé | `fuser -k 5000/tcp` puis relancer |

---

## Documentation liée

- **[DEPLOYMENT.md](DEPLOYMENT.md)** — install multi-OS, Release v8, Docker
- **[cascade/CASCADE_RULES.md](cascade/CASCADE_RULES.md)** — règles Duo / Reranker / Qwen
- **Release Cascade V8** — https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0
- **[AI_annotation](https://github.com/Cespriet/AI_annotation)** — entraînement / sources V8
