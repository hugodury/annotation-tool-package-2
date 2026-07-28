# Déploiement multi-OS / multi-PC — ISIALAB Annotation Interface

## Réponse courte

**Non**, un simple `git clone` ne suffit pas : les poids ML ne sont **pas** dans Git
(trop volumineux). Ils sont distribués via **GitHub Releases**, téléchargés
automatiquement au premier `./start.sh` / `start.ps1` / `start.bat`.

| Release | Contenu | Tag |
|---------|---------|-----|
| Modèles de base | DeBERTa-base + SBERT + cross-encoder (~1,5 Go) | [v1.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v1.0.0) |
| **Cascade V8** | MiniLM + DeBERTa Large + Reranker undetermined (~4 Go) | [v8.0.0](https://github.com/hugodury/annotation-tool-package-2/releases/tag/v8.0.0) |

Même procédure sur **Windows, macOS et Linux** — chemins relatifs `models/…`,
pas de symlink absolu machine-dépendante.

---

## Prérequis utilisateur final

| Composant | Windows | macOS | Linux |
|-----------|---------|-------|-------|
| Python 3.9+ | ✓ | ✓ | ✓ |
| Ollama | [ollama.com](https://ollama.com/) | idem | idem |
| RAM | 16 Go+ recommandés (Cascade + Qwen) ; 10 Go min. Qwen only | idem | idem |
| Disque libre (1er lancement) | **~20 Go** (venv + v1 + v8 + Qwen + marge) | idem | idem |
| Disque une fois installé | ~12–14 Go | idem | idem |

---

## Installation utilisateur (1 commande)

### Mac / Linux
```bash
git clone https://github.com/hugodury/annotation-tool-package-2.git
cd annotation-tool-package-2
chmod +x start.sh
./start.sh
```

### Windows (PowerShell)
```powershell
git clone https://github.com/hugodury/annotation-tool-package-2.git
cd annotation-tool-package-2
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

### Windows (CMD)
```bat
git clone https://github.com/hugodury/annotation-tool-package-2.git
cd annotation-tool-package-2
start.bat
```

Le script `scripts/setup.py` :
1. Crée le venv Python
2. Installe PyTorch (CPU / CUDA / MPS selon la machine)
3. Télécharge **Release v1.0.0** → `models/fine_tuned_*`
4. Télécharge **Release v8.0.0** → `models/cascade_v8/` (meta + MiniLM + DeBERTa Large + Reranker reassembled)
5. Démarre Ollama si besoin et pull Qwen 7B
6. Lance l'app sur http://127.0.0.1:5000

Après ça, les modes **Qwen only**, **DeBERTa + Qwen**, **Cascade V8 + Qwen** et
**Compare Qwen ↔ Cascade V8** fonctionnent sur n'importe quel PC.

---

## Pourquoi les modèles ne sont pas dans Git ?

| | Git | GitHub Release |
|--|--|--|
| Poids v1 (~1,5 Go) | exclus (`/models/` dans `.gitignore`) | `vldbench-models-v1.tar.gz` |
| Poids V8 (~4 Go) | exclus | assets `vldbench-cascade-v8-*` |
| Config / code | versionnés | — |

Limite GitHub **2 Go / fichier** : le Reranker (~2,1 Go) est découpé en
`.part00` / `.part01` puis réassemblé par `scripts/download_models_v8.py`.

Manifests versionnés (sans poids) :
- `models.manifest.json` — v1
- `models-v8.manifest.json` — v8 (URLs + checksums SHA-256)

---

## Cascade V8 — fonctionnement multi-OS

Après install, l'arbre est **identique** partout :

```text
models/
├── fine_tuned_deberta_base_expanded/   # Release v1
├── fine_tuned_sbert/
├── fine_tuned_cross_encoder/
└── cascade_v8/                         # Release v8
    ├── config.json
    ├── CASCADE_RULES.md
    └── models/
        ├── minilm_full_v7/model.safetensors
        ├── deberta_large_v8.1/model.safetensors
        └── reranker_undetermined_v8/model.safetensors
```

`cascade/v8_predictor.py` résout `models/cascade_v8` en chemin **relatif** au repo
(compatible Windows `\` / Linux/macOS `/`). Override optionnel :

```bash
export CASCADE_V8_ROOT=/chemin/absolu/vers/package_v8
```

DeBERTa Large tourne en **CPU** partout (NaNs observés sur GPU/MPS).

---

## LLM

| Profil | Modèle Ollama | RAM min | Usage |
|--------|---------------|---------|-------|
| qwen7b | `qwen2.5:7b-instruct` | 10 Go | LLM Qwen only / étage final V8 |

```bash
export OLLAMA_LLM_MODEL=qwen2.5:7b-instruct   # Mac/Linux
set OLLAMA_LLM_MODEL=qwen2.5:7b-instruct       # Windows CMD
```

---

## Instructions mainteneur — publier les Releases

### A) Modèles de base (v1) — déjà fait

```bash
python scripts/package_models.py
# → vldbench-models-v1.tar.gz
gh release create v1.0.0 vldbench-models-v1.tar.gz \
  --title "VLDBench models v1" \
  --notes "DeBERTa-base + SBERT + cross-encoder"
```

### B) Cascade V8 (v8.0.0)

Source des poids (local, non Git) :
`../AI_annotation/cascade_annotation_v8_complete/cascade_annotation_v8_complete/`

```bash
# 1. Créer les assets (meta + poids ; Reranker split < 2 Go)
python scripts/package_models_v8.py
# → dist/cascade-v8/ + models-v8.manifest.json à la racine

# 2. Publier la release (upload asset par asset)
python scripts/publish_models_v8.py
# ou :
# gh release create v8.0.0 dist/cascade-v8/* --repo hugodury/annotation-tool-package-2 \
#   --title "Cascade V8 models" --notes "…"
```

Assets attendus :

| Fichier | Rôle |
|---------|------|
| `vldbench-cascade-v8-meta.tar.gz` | `config.json` + tokenizers |
| `vldbench-cascade-v8-minilm.safetensors` | MiniLM |
| `vldbench-cascade-v8-deberta.safetensors` | DeBERTa Large |
| `vldbench-cascade-v8-reranker.safetensors.part00` | Reranker (partie 1) |
| `vldbench-cascade-v8-reranker.safetensors.part01` | Reranker (partie 2) |

Committer **`models-v8.manifest.json`** (checksums) dans Git — **jamais** les `.safetensors`.

### Miroir / URL custom

```bash
export MODELS_DOWNLOAD_URL=https://votre-cdn/vldbench-models-v1.tar.gz
export CASCADE_V8_DOWNLOAD_URL_META=…
export CASCADE_V8_DOWNLOAD_URL_MINILM=…
export CASCADE_V8_DOWNLOAD_URL_DEBERTA=…
export CASCADE_V8_DOWNLOAD_URL_RERANKER_PART0=…
export CASCADE_V8_DOWNLOAD_URL_RERANKER_PART1=…
```

Voir `.env.example`.

---

## Docker (optionnel)

```bash
docker compose up --build
```

- App : http://localhost:5000
- Ollama : http://localhost:11434

Les téléchargements v1 + v8 passent aussi dans le conteneur si l'espace disque le permet.

```bash
docker compose exec ollama ollama pull qwen2.5:7b-instruct
```

---

## Vérification

```bash
python scripts/system_check.py
curl http://127.0.0.1:5000/api/status
```

Attendu : `ml_models: true` (v1 **et** V8 présents).

Test rapide Cascade V8 :

```bash
./venv/bin/python - <<'PY'
from cascade.v8_predictor import CascadePredictor, default_v8_root
print("root", default_v8_root())
p = CascadePredictor()
print(p.annotate_one(
  "UK forecast second-fastest G7 growth",
  "UK worst-performing G7 economy OECD",
)["label"])
PY
```

---

## Dépannage

| Problème | Solution |
|----------|----------|
| `Modele DeBERTa manquant` | `python scripts/download_models.py` |
| Cascade V8 / Compare échoue au load | Vérifier `models/cascade_v8/.../model.safetensors` ; Release `v8.0.0` |
| Download 404 | Publier la Release ou `MODELS_DOWNLOAD_URL` / `CASCADE_V8_DOWNLOAD_URL_*` |
| Espace disque | Libérer ~20 Go au 1er install |
| Symlink `cascade_v8` cassé (ancien setup) | Supprimer le lien ; relancer `download_models_v8.py` (crée un vrai dossier) |
| Windows chemins | Ne pas hardcoder `/home/...` ; toujours chemins relatifs au repo |

---

## Checklist mainteneur avant de dire « ça marche multi-PC »

1. [ ] `/models/` dans `.gitignore`
2. [ ] Release **v1.0.0** + **v8.0.0** publiées sur GitHub
3. [ ] `models.manifest.json` + `models-v8.manifest.json` commités avec checksums
4. [ ] Test `./start.sh` sur machine **vierge** (sans symlink local)
5. [ ] Modes UI : Qwen only, DeBERTa+Qwen, Cascade V8+Qwen, Compare
