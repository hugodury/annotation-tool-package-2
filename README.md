# ISIALAB Annotation Tool

Interface web Flask pour l'annotation automatique de paires d'articles (projet **VLDBench**).

Le pipeline d'annotation repose sur une **cascade de modèles locaux** : Duo Zero Faute (MiniLM + DeBERTa Large) → Reranker undetermined → Qwen 2.5-7B via Ollama — sans aucune donnée envoyée vers un service externe.

| Ressource | Lien |
|-----------|------|
| Dépôt GitHub | <https://github.com/hugodury/annotation-tool-package-2> |
| Release modèles (v8.0.0) | <https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0> |
| Guide déploiement multi-OS | [DEPLOYMENT.md](DEPLOYMENT.md) |

---

## Démarrage rapide

> **PC neuf / clone GitHub** — une seule commande crée le venv, installe les dépendances,
> télécharge tous les modèles depuis la Release v8.0.0 et démarre l'application.
> Premier lancement : 15–45 min selon la connexion. Internet requis une seule fois.

| OS | Commande |
|----|----------|
| macOS / Linux | `chmod +x start.sh && ./start.sh` |
| Windows PowerShell | `powershell -ExecutionPolicy Bypass -File .\start.ps1` |
| Windows CMD | `start.bat` |
| Docker | `docker compose up --build` → voir [DEPLOYMENT.md](DEPLOYMENT.md) |

Ouvrir ensuite **<http://127.0.0.1:5000>** dans un navigateur.

---

## Prérequis

| Élément | Détail |
|---------|--------|
| Python | 3.9+ (3.12+ recommandé) |
| Ollama | <https://ollama.com/> — détecté et démarré automatiquement par le script |
| RAM | ≥ 16 Go recommandé (Cascade + Qwen) ; ≥ 10 Go minimum (Qwen only) |
| Disque (1er lancement) | ~16 Go libres (venv + Cascade V8 ~4–7 Go + SBERT ~0,5 Go + Qwen ~4 Go) |
| GPU | Optionnel — DeBERTa Large tourne en CPU pour des raisons de stabilité |

---

## Format JSON d'entrée

Le fichier chargé dans l'outil est une **liste de références**. Chaque référence contient un article ancre et une liste d'articles candidats (`database`) à annoter.

```json
[
  {
    "news_id": "ref_001",
    "news": "Headline or full text of the reference article",
    "topic": "Politics",
    "database": [
      { "news": "Headline or full text of a candidate article", "topic": "Politics" },
      { "news": "Another candidate article", "topic": "Politics" }
    ]
  }
]
```

> Le champ pivot lu par Run Model et l'UI est **`news`** (pas `news_title`).

Exemples disponibles dans `samples/sample1.json` … `sample4.json`.

**Labels produits** : `supporting` · `against` · `undetermined` · `not_related`
(`dismissed` = rejet manuel humain, jamais écrit par la cascade automatique).

---

## Bases de données et stockage

### 1. Fichiers JSON annotés — source principale

Les annotations sont stockées **directement dans les fichiers JSON d'entrée**, enrichis en place après chaque passage de Run Model. Aucune base relationnelle externe n'est requise pour les labels.

Chaque cible annotée reçoit les champs suivants :

| Champ | Type | Description |
|-------|------|-------------|
| `related` | str | Label final : `supporting`, `against`, `undetermined`, `not_related` |
| `similarity_annotation` | float | Score cosine SBERT v2 (0–1) |
| `similarity_source` | str | Toujours `"sbert"` |
| `cascade_route` | str | Route de décision : `v8_duo`, `v8_reranker`, `v8_qwen`, … |
| `model_confidence` | float | Confiance du modèle décideur |
| `annotated_by` | str | `"duo"`, `"reranker"` ou `"qwen"` |
| `annotated_by_label` | str | Libellé lisible du décideur |
| `cascade_v8` | dict | Détail interne : stage, règle, probabilités MiniLM/DeBERTa/Reranker |
| `llm_sim` / `llm_pred` | str/float | Proposition et similarité brutes du LLM (avant post-traitement) |
| `pipeline_compare` | dict | Résultat comparatif Qwen-only vs Cascade (mode Compare uniquement) |

Un backup horodaté est créé dans `<storage_folder>/backups/` avant chaque écriture.

### 2. `instance/annotations.db` — cache SQLite

Cache interne Flask-SQLAlchemy. **Ne contient pas les labels finaux** ; utilisé uniquement pour l'état interne de l'application (sessions en cours, etc.). Regénéré automatiquement si absent.

### 3. `instance/storage_settings.json`

Mémorise le chemin du storage folder choisi dans l'UI entre les redémarrages Flask.

```json
{ "upload_folder": "/home/user/Desktop", "updated_at": "2026-08-17T09:00:00Z" }
```

### 4. `instance/saved_sessions.json`

Registre des sessions (nom affiché, chemin du fichier JSON, date de dernière ouverture). Persistant entre redémarrages.

### 5. `instance/estimate_calibration.json`

Calibration du temps estimé Run Model, séparé par mode (`qwen_only`, `v8_qwen`). Mis à jour automatiquement après chaque run terminé.

### 6. `cascade/config.json` — configuration centrale

Seuils du pipeline, profil LLM actif, chemins des modèles, paramètres Ollama. Ne pas modifier manuellement sauf pour changer le LLM ou les seuils de décision avancés. Les variables d'environnement (`.env`) ont priorité sans modifier ce fichier.

### 7. `cascade/protocol.md` et `cascade/few_shot.json`

Prompt système P3 et exemples few-shot injectés dans chaque requête Qwen. Modifiables pour expérimenter d'autres formulations (évaluer l'impact avec le gold VLDBench avant de changer).

### 8. `logs/run_model_session.log`

Log de la session Run Model en cours. Effacé au prochain démarrage de session.

---

## Modes d'annotation

| Mode UI | Code | Comportement |
|---------|------|--------------|
| Annotate — Qwen only | `qwen_only` | Chaque paire → Qwen P3 + post-traitement |
| Annotate — Cascade (incl. Qwen) | `v8_qwen` | Duo → Reranker → Qwen (fallback) |
| Compare — Qwen ↔ Cascade | `compare` | Deux pipelines en parallèle ; accord → auto ; désaccord → revue |

### Pipeline Cascade V8 (`v8_qwen`)

```
Paire (T_ref, T_n)
  │
  ├─ Duo Zero Faute [MiniLM v7 (×0,4) + DeBERTa Large v8.1 (×0,6)]
  │     Rule 5 : désaccord MiniLM/DeBERTa  → rejet
  │     Rule 1 : P(against) > 0,35         → AUTO against     [route v8_duo]
  │     Rule 2 : P(undet) > 0,10           → rejet
  │     Rule 3 : max(P_ens) ≥ 0,98         → AUTO argmax      [route v8_duo]
  │     Rule 4 : sinon                     → rejet
  │
  ├─ Reranker undetermined (si rejet duo)
  │     P(undet) ≥ 0,70 → AUTO undetermined                   [route v8_reranker]
  │
  └─ Qwen 2.5-7B P3 + post-traitement métier                  [route v8_qwen]
```

Seuils configurables dans `models/cascade_v8/config.json`.
Règles détaillées : [`cascade/CASCADE_RULES.md`](cascade/CASCADE_RULES.md).

**Pas de revue humaine** en mode Annotate Cascade : chaque paire sort automatiquement via l'un des trois étages.

---

## Modèles ML

Les modèles ne sont **pas inclus dans le dépôt Git** (`.gitignore → /models/`).
Ils sont téléchargés automatiquement depuis la Release **v8.0.0** au premier `./start.sh`.

| Asset Release | Rôle | Taille |
|---------------|------|--------|
| `vldbench-sbert-v2.tar.gz` | SBERT v2 fine-tuné VLDBench — `similarity_annotation` | ~0,5 Go |
| `vldbench-cascade-v8-meta.tar.gz` | Config + tokenizers Cascade | < 0,1 Go |
| `vldbench-cascade-v8-minilm.safetensors` | MiniLM v7 (6 couches RoBERTa, 4 classes) | ~0,1 Go |
| `vldbench-cascade-v8-deberta.safetensors` | DeBERTa Large v8.1 (24 couches, 4 classes) | ~1,5 Go |
| `*.reranker.safetensors.part*` | Reranker XLM-RoBERTa Large binaire (version améliorée) | ~2 Go |

```
models/
├── fine_tuned_sbert/              # SBERT v2
└── cascade_v8/
    ├── config.json
    └── models/
        ├── minilm_full_v7/
        ├── deberta_large_v8.1/
        └── reranker_undetermined_v8/
```

Pour re-télécharger manuellement :
```bash
python scripts/download_models.py      # SBERT v2 + Cascade V8 (Release v8.0.0)
python scripts/download_models_v8.py   # Cascade V8 uniquement
```

---

## LLM local (Ollama)

Modèle par défaut : **`qwen2.5:7b-instruct`**

Changer de modèle :
```bash
OLLAMA_LLM_MODEL=mon-modele:tag ./start.sh
```

| Variable `.env` | Rôle |
|-----------------|------|
| `OLLAMA_HOST` | URL du serveur Ollama (défaut : `http://127.0.0.1:11434`) |
| `OLLAMA_LLM_MODEL` | Tag du modèle Ollama |
| `ANNOTATION_DATA_DIR` | Force le storage folder (verrouille l'UI) |
| `CASCADE_V8_ROOT` | Override chemin du package Cascade V8 |
| `FLASK_HOST` / `FLASK_PORT` | Bind du serveur Flask |

---

## API REST (extrait)

| Méthode | Route | Description |
|---------|-------|-------------|
| POST | `/auto_annotate` | Lancer Run Model |
| GET | `/api/auto_annotate/status` | Progression du job |
| POST | `/api/auto_annotate/estimate` | Estimation de durée |
| POST | `/api/auto_annotate/cancel` | Annuler le job |
| GET | `/api/system-check` | Checklist des prérequis |
| GET/POST | `/api/storage` | Lire / définir le storage folder |
| POST | `/api/storage/pick` | Sélecteur natif OS |
| POST | `/api/upload/pick` | Sélecteur JSON natif |
| GET | `/api/sessions` | Liste des sessions sauvegardées |
| POST | `/api/sessions/load` | Recharger une session |

---

## Structure du projet

```
annotation-tool-package-2/
├── app.py                   # Serveur Flask + routes API
├── cascade/
│   ├── __init__.py          # Point d'entrée public du package
│   ├── core.py              # Orchestration pipeline + CascadeEngine
│   ├── v8_predictor.py      # CascadePredictor : Duo + Reranker
│   ├── config.json          # Configuration centrale (seuils, LLM, chemins)
│   ├── protocol.md          # Prompt système P3 (Qwen)
│   ├── protocol_claude.md   # Variante Claude du prompt
│   ├── few_shot.json        # Exemples few-shot pour le prompt
│   ├── post_llm_regles.txt  # Règles de correction post-LLM
│   └── CASCADE_RULES.md     # Documentation des règles Duo/Reranker
├── scripts/
│   ├── annotation_store.py  # Verrous, backups, validation d'indices
│   ├── run_model_job.py     # Job Run Model (thread background, logs, reprise)
│   ├── ollama_service.py    # Démarrage Ollama et pull du LLM
│   ├── system_check.py      # Checklist prérequis (RAM, GPU, modèles, …)
│   ├── download_models.py   # Téléchargement SBERT v2 + Cascade V8
│   ├── download_models_v8.py# Téléchargement Cascade V8 uniquement
│   ├── setup.py             # Script d'installation (venv, pip, pull Qwen)
│   ├── averitec_to_cascade.py # Conversion AVeriTeC → format VLDBench
│   └── prepare_fever_submission.py # Soumission FEVER/Ev2R
├── samples/                 # JSON d'exemple (sample1–4.json)
├── templates/               # Templates HTML Jinja2 (UI en anglais)
├── models/                  # Gitignored — SBERT v2 + cascade_v8/
├── instance/                # Gitignored — DB SQLite, settings, sessions, logs
├── requirements.txt
├── docker-compose.yml
├── start.sh / start.bat / start.ps1
├── README.md
└── DEPLOYMENT.md
```

---

## Dépannage rapide

| Problème | Solution |
|----------|----------|
| Run Model grisé | Checklist `/api/system-check` : vérifier Cascade V8, SBERT, Ollama, Qwen |
| Cascade / Compare KO | `python scripts/download_models_v8.py` |
| SBERT / similarity KO | `python scripts/download_models.py` |
| RAM ○ (non bloquant) | Normal — Cascade recommande ≥ 16 Go, non bloquant |
| Pas d'estimation | Effectuer un premier Run Model avec ce mode pour calibrer |
| Fichiers hors storage | Rouvrir le JSON (copie automatique dans le storage) |
| Port 5000 occupé | `fuser -k 5000/tcp` puis relancer |

---

## Documentation associée

- **[DEPLOYMENT.md](DEPLOYMENT.md)** — installation multi-OS, Docker, Release v8.0.0
- **[cascade/CASCADE_RULES.md](cascade/CASCADE_RULES.md)** — règles Duo / Reranker / Qwen
- **Release v8.0.0** — <https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0>
