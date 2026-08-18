"""Serveur Flask — interface web d'annotation VLDBench.

Ce module expose l'API REST et les routes HTML de l'outil d'annotation.
Il orchestre :
- Le chargement paresseux des modèles ML (SBERT, CascadeEngine).
- La gestion du storage folder (JSON annotés, backups).
- Les sessions sauvegardées (``instance/saved_sessions.json``).
- Le lancement et le suivi des jobs Run Model (thread background).
- Les sélecteurs natifs OS (zenity / Finder / PowerShell).

Bases de données / stockage :
    instance/annotations.db
        Cache SQLite (Flask-SQLAlchemy). Utilisé uniquement pour le cache
        interne de l'app, pas pour stocker les labels finaux.
    instance/storage_settings.json
        Chemin du storage folder choisi par l'utilisateur.
    instance/saved_sessions.json
        Registre des sessions (nom, fichier, date) persistant entre redémarrages.
    instance/estimate_calibration.json
        Calibration du temps estimé Run Model par mode (qwen_only / v8_qwen).
    <storage_folder>/<fichier>.json
        Fichiers JSON annotés (format VLDBench : liste de références + database).
        Chaque écriture est précédée d'une copie dans ``backups/``.
    logs/run_model_session.log
        Log de la session Run Model en cours.
"""
from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import json
import os
import sys
import time
import threading
import shutil
from datetime import datetime
import numpy as np
import torch
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
STORAGE_SETTINGS_PATH = BASE_DIR / "instance" / "storage_settings.json"
SAVED_SESSIONS_PATH = BASE_DIR / "instance" / "saved_sessions.json"
STORAGE_HISTORY_PATH = BASE_DIR / "instance" / "storage_history.json"
DEFAULT_UPLOAD_DIR = BASE_DIR / "uploads"
load_dotenv(BASE_DIR / ".env")
sys.path.insert(0, str(BASE_DIR / "scripts"))
from annotation_store import backup_json_file, file_lock_for, parse_index_range  # noqa: E402
from native_dialogs import pick_folder, pick_json_file, dialog_capabilities, picker_unavailable_message  # noqa: E402
from ollama_service import (  # noqa: E402
    ensure_ollama_ready,
    ensure_ollama_ready_async,
    ensure_state,
)
from system_check import build_report  # noqa: E402
from run_model_job import (  # noqa: E402
    count_range_annotation_stats,
    estimate_batch,
    find_resume_index,
    get_job_status,
    get_session_log_path,
    is_job_running,
    models_warm_in_session,
    prepare_session_logs,
    request_cancel_batch,
    start_batch,
)
from cascade.core import normalize_cascade_mode  # noqa: E402

# Lazy loaded models
sbert_model = None
cascade_engine = None
_dialog_lock = threading.Lock()


def _format_elapsed(seconds: float) -> str:
    """Formate une durée en secondes en chaîne lisible (ex. ``2 h 3 min 5 s``).

    Args:
        seconds: Durée en secondes (float).

    Returns:
        Chaîne formatée.
    """
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    m, rem = divmod(s, 60)
    if m < 60:
        return f"{m} min {rem} s" if rem else f"{m} min"
    h, rem_m = divmod(m, 60)
    tail = f" {rem_m} min" if rem_m else ""
    return f"{h} h{tail}" + (f" {rem} s" if rem else "")


def _elapsed_info(batch_start: float) -> dict:
    """Retourne le temps écoulé depuis ``batch_start`` en secondes et en label.

    Args:
        batch_start: Timestamp de départ (``time.monotonic()``).

    Returns:
        Dict avec ``elapsed_seconds`` (float) et ``elapsed_label`` (str).
    """
    elapsed = round(time.monotonic() - batch_start, 1)
    return {"elapsed_seconds": elapsed, "elapsed_label": _format_elapsed(elapsed)}


def get_cascade_engine():
    """Retourne (en le créant si nécessaire) l'instance globale de CascadeEngine.

    Chargement paresseux : les modèles MiniLM, DeBERTa et Reranker ne sont
    chargés qu'au premier appel. Thread-safe via le GIL Python.

    Returns:
        Instance unique de ``cascade.core.CascadeEngine``.
    """
    global cascade_engine
    if cascade_engine is None:
        from cascade.core import CascadeEngine
        cascade_engine = CascadeEngine()
    return cascade_engine


def get_sbert_model():
    """Retourne (en le créant si nécessaire) l'instance globale de SentenceTransformer.

    Charge ``models/fine_tuned_sbert/`` (SBERT v2 fine-tuned VLDBench).
    Utilisé exclusivement pour le calcul de ``similarity_annotation`` (cosine).

    Returns:
        Instance unique de ``sentence_transformers.SentenceTransformer``.
    """
    global sbert_model
    if sbert_model is None:
        from sentence_transformers import SentenceTransformer
        model_path = BASE_DIR / 'models' / 'fine_tuned_sbert'
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        sbert_model = SentenceTransformer(str(model_path), device=device)
    return sbert_model

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{(BASE_DIR / "instance" / "annotations.db").as_posix()}'


def _storage_locked_by_env() -> bool:
    return bool(os.environ.get("ANNOTATION_DATA_DIR", "").strip())


def _load_storage_settings() -> dict:
    if not STORAGE_SETTINGS_PATH.is_file():
        return {}
    try:
        with open(STORAGE_SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_storage_settings(upload_folder: Path) -> None:
    STORAGE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORAGE_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "upload_folder": str(upload_folder),
                "updated_at": datetime.utcnow().isoformat() + "Z",
            },
            f,
            indent=2,
        )


def _resolve_storage_path(raw: str) -> Path:
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def _default_upload_dir() -> Path:
    env_path = os.environ.get("ANNOTATION_DATA_DIR", "").strip()
    if env_path:
        return _resolve_storage_path(env_path)
    saved = _load_storage_settings().get("upload_folder", "").strip()
    if saved:
        return _resolve_storage_path(saved)
    return DEFAULT_UPLOAD_DIR.resolve()


def _storage_source() -> str:
    if _storage_locked_by_env():
        return "env"
    if _load_storage_settings().get("upload_folder", "").strip():
        return "settings"
    return "default"


def _load_storage_history() -> list[str]:
    paths = [str(DEFAULT_UPLOAD_DIR.resolve())]
    if STORAGE_HISTORY_PATH.is_file():
        try:
            with open(STORAGE_HISTORY_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                paths.extend(str(p) for p in data if p)
        except (OSError, json.JSONDecodeError):
            pass
    try:
        current = str(Path(app.config["UPLOAD_FOLDER"]).resolve())
        paths.append(current)
    except (KeyError, OSError):
        pass
    unique: list[str] = []
    seen = set()
    for raw in paths:
        try:
            key = str(Path(raw).expanduser().resolve())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _register_storage_folder(path: Path) -> None:
    try:
        resolved = str(path.expanduser().resolve())
    except OSError:
        return
    history = []
    if STORAGE_HISTORY_PATH.is_file():
        try:
            with open(STORAGE_HISTORY_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                history = [str(p) for p in data if p]
        except (OSError, json.JSONDecodeError):
            history = []
    if resolved not in history:
        history.insert(0, resolved)
    STORAGE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORAGE_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history[:50], f, indent=2)


def _set_upload_folder(raw: str) -> Path:
    old = app.config.get("UPLOAD_FOLDER")
    path = _resolve_storage_path(raw)
    if path.exists() and not path.is_dir():
        raise ValueError("Path exists but is not a folder.")
    path.mkdir(parents=True, exist_ok=True)
    test_file = path / ".write_test"
    try:
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
    except OSError as exc:
        raise ValueError(f"Folder is not writable: {exc}") from exc
    (path / "backups").mkdir(parents=True, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = str(path)
    if not _storage_locked_by_env():
        _save_storage_settings(path)
    _register_storage_folder(path)
    if old:
        _register_storage_folder(Path(old))
    return path


def _path_in_storage(path: Path) -> bool:
    """True only if the file sits directly in the storage folder (not a subfolder).

    Choosing Desktop as storage must not treat Desktop/Stage/.../uploads/foo.json
    as already stored — otherwise annotations keep writing into nested uploads/.
    """
    storage = Path(app.config["UPLOAD_FOLDER"]).resolve()
    try:
        return path.expanduser().resolve().parent == storage
    except OSError:
        return False


def _materialize_into_storage(source: Path) -> Path:
    """Copy JSON into the storage folder unless it already lives there.

    Annotations always write to the active session path; without this, Choose JSON
    kept editing the original file (e.g. …/uploads/) even when storage is Desktop.
    """
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"File not found: {source}")
    if _path_in_storage(source):
        return source
    storage = Path(app.config["UPLOAD_FOLDER"]).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    dest = (storage / source.name).resolve()
    if dest != source:
        shutil.copy2(source, dest)
    return dest


def _init_upload_folder() -> Path:
    path = _default_upload_dir()
    path.mkdir(parents=True, exist_ok=True)
    (path / "backups").mkdir(parents=True, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = str(path)
    _register_storage_folder(path)
    _register_storage_folder(DEFAULT_UPLOAD_DIR)
    return path


_init_upload_folder()
db = SQLAlchemy(app)
os.makedirs(BASE_DIR / 'instance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('logs', exist_ok=True)


class AnnotationFile(db.Model):
    """Fichier JSON charge dans l'outil."""
    __tablename__ = 'annotation_files'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), unique=True, nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ReferenceAnnotation(db.Model):
    """Etat d'annotation d'une reference (news_id) dans un fichier donne."""
    __tablename__ = 'reference_annotations'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False, index=True)
    news_id = db.Column(db.String(50), nullable=False, index=True)
    status = db.Column(db.String(20), nullable=False, default='pending')  # pending, partial, complete
    source = db.Column(db.String(20), default='import')  # manual, auto, import
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('filename', 'news_id', name='uq_ref_file_news'),)


class TargetAnnotation(db.Model):
    """Annotation detaillee de chaque cible dans une reference."""
    __tablename__ = 'target_annotations'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False, index=True)
    news_id = db.Column(db.String(50), nullable=False, index=True)
    target_index = db.Column(db.Integer, nullable=False)
    related = db.Column(db.String(50))
    similarity_annotation = db.Column(db.Float)
    cascade_route = db.Column(db.String(50))
    source = db.Column(db.String(20), default='import')
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (
        db.UniqueConstraint('filename', 'news_id', 'target_index', name='uq_target_file_news_idx'),
    )


# Ancien modele conserve pour migration depuis annotations.db existantes.
class ProcessedNews(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    news_id = db.Column(db.String(50), unique=True, nullable=False)
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, news_id):
        self.news_id = news_id


def _target_is_annotated(target: dict) -> bool:
    rel = target.get('related')
    if rel is None or str(rel).strip() == '':
        return False
    if str(rel).strip().lower() == 'dismissed':
        return True
    sim = target.get('similarity_annotation')
    if sim is None or sim == '':
        return False
    try:
        float(sim)
        return True
    except (TypeError, ValueError):
        return False


def reference_status_from_item(item: dict) -> str:
    targets = item.get('database') or []
    if not targets:
        return 'pending'
    annotated = sum(1 for t in targets if _target_is_annotated(t))
    if annotated == 0:
        return 'pending'
    if annotated == len(targets):
        return 'complete'
    return 'partial'


def _upsert_annotation_file(filename: str) -> None:
    record = AnnotationFile.query.filter_by(filename=filename).first()
    if record is None:
        db.session.add(AnnotationFile(filename=filename))
    else:
        record.updated_at = datetime.utcnow()


def sync_reference_to_db(filename: str, item: dict, source: str = 'import') -> str:
    news_id = str(item.get('news_id', ''))
    status = reference_status_from_item(item)
    record = ReferenceAnnotation.query.filter_by(filename=filename, news_id=news_id).first()
    if record is None:
        record = ReferenceAnnotation(filename=filename, news_id=news_id)
        db.session.add(record)
    record.status = status
    record.source = source
    record.updated_at = datetime.utcnow()

    for idx, target in enumerate(item.get('database') or []):
        if not _target_is_annotated(target):
            continue
        ta = TargetAnnotation.query.filter_by(
            filename=filename, news_id=news_id, target_index=idx
        ).first()
        if ta is None:
            ta = TargetAnnotation(filename=filename, news_id=news_id, target_index=idx)
            db.session.add(ta)
        ta.related = str(target.get('related', ''))
        sim = target.get('similarity_annotation')
        ta.similarity_annotation = float(sim) if sim is not None and sim != '' else None
        ta.cascade_route = target.get('cascade_route')
        ta.source = source
        ta.updated_at = datetime.utcnow()
    return status


def sync_file_annotations(filename: str, data: list, source: str = 'import') -> None:
    _upsert_annotation_file(filename)
    for item in data:
        sync_reference_to_db(filename, item, source=source)


def enrich_data_with_status(filename: str, data: list) -> list:
    for item in data:
        content_status = reference_status_from_item(item)
        item['annotation_status'] = content_status
        item['is_processed'] = content_status == 'complete'
    return data


def get_processed_ids_for_file(filename: str) -> set[str]:
    rows = ReferenceAnnotation.query.filter_by(filename=filename, status='complete').all()
    return {row.news_id for row in rows}


def _migrate_legacy_db() -> None:
    from sqlalchemy import inspect, text

    db.create_all()
    inspector = inspect(db.engine)
    if 'processed_news' not in inspector.get_table_names():
        return
    if ReferenceAnnotation.query.count() > 0:
        return
    try:
        legacy_rows = db.session.execute(text('SELECT news_id FROM processed_news')).fetchall()
        for (news_id,) in legacy_rows:
            db.session.add(
                ReferenceAnnotation(
                    filename='_legacy_',
                    news_id=str(news_id),
                    status='complete',
                    source='legacy',
                )
            )
        db.session.commit()
    except Exception:
        db.session.rollback()


with app.app_context():
    _migrate_legacy_db()


def _load_cascade_config() -> dict:
    from cascade.core import load_config

    return load_config()


# Start Ollama + LLM in background; fresh Run Model log for this server session.
prepare_session_logs(BASE_DIR / "logs")
ensure_ollama_ready_async(_load_cascade_config(), BASE_DIR / "cascade" / "config.json")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def api_health():
    return jsonify({'status': 'ok'})


@app.route('/api/status')
def api_status():
    from cascade.core import load_config
    cfg = load_config()
    report = build_report(BASE_DIR, cfg)
    report["ollama_ensure"] = ensure_state()
    return jsonify(report)


@app.route('/api/ensure-ollama', methods=['POST'])
def api_ensure_ollama():
    """Démarre Ollama et télécharge le LLM si nécessaire (appel synchrone)."""
    cfg_path = BASE_DIR / "cascade" / "config.json"
    cfg = _load_cascade_config()
    try:
        ensure_ollama_ready(cfg, cfg_path, install_binary=True, pull_llm=True, quiet=True)
        report = build_report(BASE_DIR, cfg)
        report["ollama_ensure"] = ensure_state()
        return jsonify({"ok": True, "report": report})
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e), "ollama_ensure": ensure_state()}), 503


@app.route('/api/system-check')
def api_system_check():
    """Alias explicite pour la vérification machine (même payload que /api/status)."""
    return api_status()


def _save_json_to_upload_folder(raw: str, safe_name: str) -> str:
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    lock = file_lock_for(upload_path)
    with lock:
        with open(upload_path, 'w', encoding='utf-8') as f:
            f.write(raw)
    return upload_path


def _import_json_payload(raw: str, safe_name: str, source_path: Path | None = None) -> dict:
    raw = raw.replace(': NaN', ': null')
    data = json.loads(raw)
    if source_path is not None:
        _set_current_session(source_path)
        safe_name = source_path.name
    else:
        upload_path = Path(app.config['UPLOAD_FOLDER']) / safe_name
        _set_current_session(upload_path)
    if isinstance(data, dict):
        data = [data]
    sync_file_annotations(safe_name, data, source='import')
    db.session.commit()
    data = enrich_data_with_status(safe_name, data)
    processed_ids = get_processed_ids_for_file(safe_name)
    return {
        'data': data,
        'processed_ids': list(processed_ids),
        'filename': safe_name,
        'path': app.config.get('current_file_path'),
    }


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.json'):
        try:
            safe_name = secure_filename(file.filename) or 'upload.json'
            file.seek(0)
            raw = file.read().decode('utf-8')
            _save_json_to_upload_folder(raw, safe_name)
            return jsonify(_import_json_payload(raw, safe_name))
        except Exception as e:
            return jsonify({'error': f'Invalid JSON file: {str(e)}'}), 400
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/upload/pick', methods=['POST'])
def api_upload_pick():
    if is_job_running():
        return jsonify({'error': 'Cannot load a file while Run Model is running.'}), 409
    if not _dialog_lock.acquire(blocking=False):
        return jsonify({'error': 'A system dialog is already open.'}), 409
    try:
        picked = pick_json_file(app.config['UPLOAD_FOLDER'])
    finally:
        _dialog_lock.release()
    if not picked:
        reason = picker_unavailable_message()
        if reason:
            return jsonify({'error': reason, 'dialog_unavailable': True}), 503
        return jsonify({'cancelled': True})
    source = Path(picked)
    if source.suffix.lower() != '.json':
        return jsonify({'error': 'Please choose a .json file.'}), 400
    try:
        working = _materialize_into_storage(source)
        raw = working.read_text(encoding='utf-8')
        safe_name = secure_filename(working.name) or 'upload.json'
        return jsonify(_import_json_payload(raw, safe_name, source_path=working))
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Invalid JSON file: {e}'}), 400

@app.route('/resume', methods=['GET'])
def resume_session():
    # Find the most recently modified json file in uploads
    upload_dir = app.config['UPLOAD_FOLDER']
    files = [f for f in os.listdir(upload_dir) if f.endswith('.json')]
    if not files:
        return jsonify({'error': 'No previous session found in uploads folder.'}), 404
        
    files.sort(key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)), reverse=True)
    latest_file = files[0]
    app.config['current_filename'] = latest_file
    
    file_path = os.path.join(upload_dir, latest_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
        
    sync_file_annotations(latest_file, data, source='import')
    db.session.commit()
    data = enrich_data_with_status(latest_file, data)
    processed_ids = get_processed_ids_for_file(latest_file)
        
    return jsonify({'data': data, 'processed_ids': list(processed_ids), 'filename': latest_file})

@app.route('/download', methods=['GET'])
def download_file():
    file_path = _resolve_current_file_path()
    if not file_path:
        return jsonify({'error': 'No file to download'}), 400
    return send_file(
        file_path,
        as_attachment=True,
        download_name=f"annotated_{file_path.name}",
    )

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    req = request.json or {}
    news_id = str(req.get('news_id', ''))
    annotation = req.get('annotation')
    
    if not news_id:
        return jsonify({'error': 'Missing news_id.'}), 400
    if not isinstance(annotation, list):
        return jsonify({'error': 'Invalid annotation (expected a list).'}), 400
    
    file_path = _resolve_current_file_path()
    if not file_path:
        return jsonify({'error': 'Original file not found. Please load a JSON file.'}), 400
    original_filename = file_path.name
    # Load the file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
        
    # Update the annotation for the correct news_id
    updated_item = None
    for item in data:
        if str(item.get('news_id')) == news_id:
            db_list = item.get('database', [])
            if len(annotation) != len(db_list):
                return jsonify({
                    'error': (
                        f'Wrong number of targets: {len(annotation)} sent, '
                        f'{len(db_list)} expected.'
                    ),
                }), 400
                for i, ann in enumerate(annotation):
                    db_list[i]['similarity_annotation'] = ann.get('similarity')
                    db_list[i]['related'] = ann.get('relation')
            updated_item = item
            break
    if updated_item is None:
        return jsonify({'error': 'News ID not found in file'}), 400

    lock = file_lock_for(file_path)
    with lock:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    sync_reference_to_db(original_filename, updated_item, source='manual')
        db.session.commit()
    status = reference_status_from_item(updated_item)
    return jsonify({
        'message': 'Annotation saved successfully',
        'annotation_status': status,
        'is_processed': status == 'complete',
    })

@app.route('/clear_database', methods=['POST'])
def clear_database():
    try:
        TargetAnnotation.query.delete()
        ReferenceAnnotation.query.delete()
        AnnotationFile.query.delete()
        ProcessedNews.query.delete()
        db.session.commit()
        return jsonify({'message': 'Database cleared successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def _load_saved_sessions_registry() -> list[dict]:
    SAVED_SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sessions: list[dict] = []
    if SAVED_SESSIONS_PATH.is_file():
        try:
            with open(SAVED_SESSIONS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                sessions = [s for s in data if isinstance(s, dict) and s.get("path")]
        except (OSError, json.JSONDecodeError):
            sessions = []
    legacy_path = BASE_DIR / "instance" / "known_sessions.json"
    if legacy_path.is_file():
        try:
            with open(legacy_path, encoding="utf-8") as f:
                legacy = json.load(f)
            if isinstance(legacy, list):
                for raw in legacy:
                    if raw:
                        sessions.append({"path": str(raw)})
        except (OSError, json.JSONDecodeError):
            pass
        if sessions:
            _save_saved_sessions_registry(sessions)
            try:
                legacy_path.unlink()
            except OSError:
                pass
    return sessions


def _save_saved_sessions_registry(sessions: list[dict]) -> None:
    SAVED_SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    unique: list[dict] = []
    seen: set[str] = set()
    for session in sessions:
        raw = session.get("path")
        if not raw:
            continue
        try:
            key = str(Path(raw).expanduser().resolve())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(session)
    with open(SAVED_SESSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(unique[:500], f, indent=2)


def _session_record(path: Path) -> dict | None:
    try:
        resolved = path.expanduser().resolve()
        stat = resolved.stat()
    except OSError:
        return None
    if not resolved.is_file() or resolved.suffix.lower() != ".json":
        return None
    return {
        "path": str(resolved),
        "filename": resolved.name,
        "parent": str(resolved.parent),
        "last_used": datetime.utcnow().isoformat() + "Z",
        "mtime": stat.st_mtime,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
    }


def _register_saved_session(path: Path) -> None:
    entry = _session_record(path)
    if not entry:
        return
    sessions = _load_saved_sessions_registry()
    kept = [s for s in sessions if s.get("path") != entry["path"]]
    kept.insert(0, entry)
    _save_saved_sessions_registry(kept[:500])


def _list_saved_sessions() -> list[dict]:
    sessions = _load_saved_sessions_registry()
    out: list[dict] = []
    seen: set[str] = set()
    for session in sessions:
        raw = session.get("path")
        if not raw:
            continue
        try:
            path = Path(raw).expanduser().resolve()
            key = str(path)
        except OSError:
            continue
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        fresh = _session_record(path)
        if fresh:
            fresh["last_used"] = session.get("last_used") or fresh["last_used"]
            out.append(fresh)
    out.sort(key=lambda s: s.get("last_used", ""), reverse=True)
    return out


def _remove_saved_session(path: Path) -> None:
    try:
        target = str(path.expanduser().resolve())
    except OSError:
        return
    kept = [s for s in _load_saved_sessions_registry() if s.get("path") != target]
    _save_saved_sessions_registry(kept)


def _set_current_session(file_path: Path) -> str:
    resolved = file_path.expanduser().resolve()
    app.config["current_filename"] = resolved.name
    app.config["current_file_path"] = str(resolved)
    _register_saved_session(resolved)
    return resolved.name


def _clear_current_session() -> None:
    app.config["current_filename"] = None
    app.config.pop("current_file_path", None)


def _resolve_session_path(filename: str = "", path: str = "") -> Path | None:
    raw_path = (path or "").strip()
    if raw_path:
        try:
            candidate = Path(raw_path).expanduser().resolve()
        except OSError:
            return None
        if candidate.is_file() and candidate.suffix.lower() == ".json":
            return candidate
    safe = _safe_upload_path(filename)
    if safe:
        return safe
    current = app.config.get("current_file_path")
    if current and filename and Path(current).name == Path(filename).name:
        try:
            candidate = Path(current).expanduser().resolve()
            if candidate.is_file():
                return candidate
        except OSError:
            pass
    if not filename:
        return None
    name = Path(filename).name
    for session in _list_saved_sessions():
        if session.get("filename") == name:
            try:
                candidate = Path(session["path"]).resolve()
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
    return None


def _resolve_current_file_path() -> Path | None:
    explicit = app.config.get("current_file_path")
    if explicit:
        try:
            candidate = Path(explicit).expanduser().resolve()
            if candidate.is_file():
                return candidate
        except OSError:
            pass
    filename = app.config.get("current_filename")
    if filename:
        upload_candidate = Path(app.config["UPLOAD_FOLDER"]) / filename
        if upload_candidate.is_file():
            return upload_candidate.resolve()
        return _resolve_session_path(filename=filename)
    return None


def _safe_upload_path(filename: str) -> Path | None:
    """Chemin JSON dans uploads/ sans alterer le nom (espaces, parentheses)."""
    if not filename or not isinstance(filename, str):
        return None
    name = Path(filename).name
    if not name.endswith('.json') or name in ('.', '..'):
        return None
    upload_dir = Path(app.config['UPLOAD_FOLDER']).resolve()
    candidate = (upload_dir / name).resolve()
    try:
        candidate.relative_to(upload_dir)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _resolve_current_filename() -> str | None:
    file_path = _resolve_current_file_path()
    if file_path:
        return file_path.name
        upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.isdir(upload_dir):
        return None
        files = [f for f in os.listdir(upload_dir) if f.endswith('.json')]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)), reverse=True)
        latest = files[0]
        app.config['current_filename'] = latest
        app.config['current_file_path'] = str(Path(upload_dir) / latest)
        return latest
    return None


def _load_upload_data(filename: str | None = None) -> tuple[list, Path]:
    file_path = _resolve_current_file_path()
    if file_path is None and filename:
        file_path = _resolve_session_path(filename=filename)
    if file_path is None:
        raise FileNotFoundError("No JSON file loaded.")
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    _set_current_session(file_path)
    return data, file_path


@app.route('/api/dialogs/capabilities')
def api_dialog_capabilities():
    return jsonify(dialog_capabilities())


@app.route('/api/storage')
def api_storage_get():
    folder = Path(app.config['UPLOAD_FOLDER']).resolve()
    return jsonify({
        'path': str(folder),
        'default_path': str(DEFAULT_UPLOAD_DIR.resolve()),
        'home_path': str(Path.home().resolve()),
        'writable': os.access(folder, os.W_OK),
        'source': _storage_source(),
        'locked': _storage_locked_by_env(),
    })


@app.route('/api/storage/pick', methods=['POST'])
def api_storage_pick():
    if is_job_running():
        return jsonify({'error': 'Cannot change storage folder while Run Model is running.'}), 409
    if _storage_locked_by_env():
        return jsonify({
            'error': 'Storage folder is locked by ANNOTATION_DATA_DIR in .env.',
            'locked': True,
        }), 403
    if not _dialog_lock.acquire(blocking=False):
        return jsonify({'error': 'A system dialog is already open.'}), 409
    try:
        picked = pick_folder(app.config['UPLOAD_FOLDER'])
    finally:
        _dialog_lock.release()
    if not picked:
        reason = picker_unavailable_message()
        if reason:
            return jsonify({'error': reason, 'dialog_unavailable': True}), 503
        return jsonify({'cancelled': True})
    try:
        folder = _set_upload_folder(picked)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    app.config['current_filename'] = None
    app.config.pop('current_file_path', None)
    return jsonify({
        'ok': True,
        'path': str(folder),
        'cancelled': False,
        'message': f'Storage folder: {folder}',
    })


@app.route('/api/storage', methods=['POST'])
def api_storage_set():
    if is_job_running():
        return jsonify({'error': 'Cannot change storage folder while Run Model is running.'}), 409
    if _storage_locked_by_env():
        return jsonify({
            'error': 'Storage folder is fixed by ANNOTATION_DATA_DIR in .env.',
            'path': app.config['UPLOAD_FOLDER'],
            'locked': True,
        }), 403
    req = request.json or {}
    raw = (req.get('path') or '').strip()
    if not raw:
        return jsonify({'error': 'Missing folder path.'}), 400
    try:
        folder = _set_upload_folder(raw)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    app.config['current_filename'] = None
    app.config.pop('current_file_path', None)
    return jsonify({
        'ok': True,
        'path': str(folder),
        'message': f'Storage folder updated: {folder}',
        'source': _storage_source(),
    })


@app.route('/api/sessions')
def api_sessions():
    storage_dir = Path(app.config['UPLOAD_FOLDER']).resolve()
    current_file = _resolve_current_file_path()
    if current_file:
        _register_saved_session(current_file)
    sessions = _list_saved_sessions()
    current_path = app.config.get('current_file_path')
    current_name = app.config.get('current_filename')
    for session in sessions:
        try:
            session['in_storage_folder'] = (
                Path(session['path']).resolve().parent == storage_dir
            )
        except OSError:
            session['in_storage_folder'] = False
    return jsonify({
        'sessions': sessions,
        'current': current_name,
        'current_path': current_path,
        'storage_folder': str(storage_dir),
        'scope': 'saved',
    })


@app.route('/api/sessions/load', methods=['POST'])
def api_load_session():
    req = request.json or {}
    file_path = _resolve_session_path(
        filename=req.get('filename', ''),
        path=req.get('path', ''),
    )
    if not file_path:
        return jsonify({'error': 'File not found.'}), 404
    try:
        file_path = _materialize_into_storage(file_path)
    except OSError as exc:
        return jsonify({'error': f'Could not copy into storage folder: {exc}'}), 500
    filename = _set_current_session(file_path)
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    sync_file_annotations(filename, data, source='import')
    db.session.commit()
    data = enrich_data_with_status(filename, data)
    processed_ids = get_processed_ids_for_file(filename)
    return jsonify({
        'data': data,
        'processed_ids': list(processed_ids),
        'filename': filename,
        'path': str(file_path),
    })


@app.route('/api/sessions/delete', methods=['POST'])
def api_delete_session():
    if is_job_running():
        return jsonify({'error': 'Cannot remove session while Run Model is running.'}), 409
    req = request.json or {}
    mode = (req.get('mode') or 'list').strip().lower()
    if mode not in ('list', 'disk'):
        return jsonify({'error': 'Invalid mode. Use "list" or "disk".'}), 400
    raw_path = (req.get('path') or '').strip()
    filename = Path(req.get('filename', '') or '').name
    target_path = ''
    file_path: Path | None = None
    if raw_path:
        try:
            target_path = str(Path(raw_path).expanduser().resolve())
            file_path = Path(target_path)
        except OSError:
            target_path = raw_path
    elif filename:
        file_path = _resolve_session_path(filename=filename, path='')
        if file_path:
            target_path = str(file_path.resolve())
    if not target_path:
        return jsonify({'error': 'Session not found.'}), 404

    registry = _load_saved_sessions_registry()
    if not any(s.get('path') == target_path for s in registry):
        return jsonify({'error': 'Session not in saved list.'}), 404

    if mode == 'list':
        _remove_saved_session(Path(target_path))
        if app.config.get('current_file_path') == target_path:
            _clear_current_session()
        elif filename and app.config.get('current_filename') == filename:
            _clear_current_session()
        label = filename or Path(target_path).name
        return jsonify({
            'ok': True,
            'mode': 'list',
            'message': f'Retiré de la liste : {label} (fichier conservé sur l\'ordinateur)',
        })

    if not file_path or not file_path.is_file():
        return jsonify({'error': 'File not found on disk.'}), 404
    try:
        ReferenceAnnotation.query.filter_by(filename=filename or file_path.name).delete()
        TargetAnnotation.query.filter_by(filename=filename or file_path.name).delete()
        AnnotationFile.query.filter_by(filename=filename or file_path.name).delete()
        file_path.unlink()
        _remove_saved_session(file_path)
        if app.config.get('current_file_path') == target_path:
            _clear_current_session()
        elif filename and app.config.get('current_filename') == filename:
            _clear_current_session()
        db.session.commit()
        label = filename or file_path.name
        return jsonify({
            'ok': True,
            'mode': 'disk',
            'message': f'Fichier supprimé de l\'ordinateur : {label}',
        })
    except OSError as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/resync', methods=['POST'])
def api_resync():
    filename = _resolve_current_filename()
    if not filename:
        return jsonify({'error': 'No file loaded.'}), 400
    data, _ = _load_upload_data(filename)
    sync_file_annotations(filename, data, source='import')
    db.session.commit()
    data = enrich_data_with_status(filename, data)
    processed_ids = get_processed_ids_for_file(filename)
    return jsonify({
        'message': 'Database resynced from JSON.',
        'data': data,
        'processed_ids': list(processed_ids),
        'filename': filename,
    })


@app.route('/api/logs/run_model/latest')
def api_latest_run_log():
    logs_dir = BASE_DIR / 'logs'
    session_path = get_session_log_path()
    latest = session_path if session_path and session_path.is_file() else None
    if latest is None and logs_dir.is_dir():
        files = sorted(
            logs_dir.glob('run_model_*.log'),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        latest = files[0] if files else None
    if latest is None:
        return jsonify({'filename': None, 'content': ''})
    content = latest.read_text(encoding='utf-8', errors='replace')
    if len(content) > 12000:
        content = content[-12000:]
    return jsonify({'filename': latest.name, 'content': content})


@app.route('/api/logs/clear', methods=['POST'])
def api_clear_logs():
    if is_job_running():
        return jsonify({'error': 'Cannot clear logs while Run Model is running.'}), 409
    prepare_session_logs(BASE_DIR / "logs")
    return jsonify({'ok': True, 'message': 'Run Model logs cleared.'})


@app.route('/api/auto_annotate/cancel', methods=['POST'])
def api_auto_annotate_cancel():
    if request_cancel_batch():
        return jsonify({'cancelled': True, 'message': 'Cancellation requested…'})
    return jsonify({'error': 'No Run Model batch in progress.'}), 409


@app.route('/api/auto_annotate/status')
def api_auto_annotate_status():
    return jsonify(get_job_status())


@app.route('/api/auto_annotate/estimate', methods=['POST'])
def api_auto_annotate_estimate():
    req = request.json or {}
    original_filename = _resolve_current_filename()
    if not original_filename:
        return jsonify({'error': 'No file loaded.'}), 400
    data, _ = _load_upload_data(original_filename)
    try:
        start_index, end_index = parse_index_range(req, len(data))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    force_reannotate = bool(req.get('force_reannotate', False))
    cascade_mode = normalize_cascade_mode(req.get('cascade_mode'))
    cfg = _load_cascade_config()
    report = build_report(BASE_DIR, cfg)
    device = report.get('hardware', {}).get('gpu', {}).get('device', 'cpu')
    is_cpu = device == 'cpu'
    est = estimate_batch(
        data,
        start_index,
        end_index,
        is_cpu=is_cpu,
        cfg=cfg,
        ref_status_fn=reference_status_from_item,
        target_is_annotated_fn=_target_is_annotated,
        logs_dir=BASE_DIR / "logs",
        force_reannotate=force_reannotate,
        device_type=device,
        models_warm=models_warm_in_session(),
        base_dir=BASE_DIR,
        filename=original_filename,
        cascade_mode=cascade_mode,
    )
    return jsonify(est)


@app.route('/api/auto_annotate/resume', methods=['GET'])
def api_auto_annotate_resume():
    original_filename = _resolve_current_filename()
    if not original_filename:
        return jsonify({'error': 'No file loaded.'}), 400
    data, _ = _load_upload_data(original_filename)
    try:
        start_index = int(request.args.get('start_index', 0))
        end_index = int(request.args.get('end_index', 0))
        if start_index < 0 or end_index >= len(data) or start_index > end_index:
            raise ValueError("Plage d'index invalide.")
    except (TypeError, ValueError) as e:
        return jsonify({'error': str(e)}), 400
    hint = find_resume_index(data, start_index, end_index, reference_status_from_item)
    return jsonify(hint)


@app.route('/auto_annotate', methods=['POST'])
def auto_annotate():
    if is_job_running():
        return jsonify({'error': 'A Run Model batch is already running.'}), 409

    req = request.json or {}
    try:
        threshold = float(req.get('threshold', 0.95))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid threshold value.'}), 400

    original_filename = _resolve_current_filename()
    if not original_filename:
        return jsonify({'error': 'File not found. Please re-upload your JSON.'}), 400

    data, file_path = _load_upload_data(original_filename)
    try:
        start_index, end_index = parse_index_range(req, len(data))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    force_reannotate = bool(req.get('force_reannotate', False))
    cascade_mode = normalize_cascade_mode(req.get('cascade_mode'))
    range_stats = count_range_annotation_stats(
        data,
        start_index,
        end_index,
        ref_status_fn=reference_status_from_item,
        target_is_annotated_fn=_target_is_annotated,
    )
    if range_stats['has_existing_annotations'] and not force_reannotate:
        if range_stats['all_annotated']:
            return jsonify({
                'error': 'This range is already fully annotated.',
                'needs_confirmation': True,
                'message': (
                    f"All targets are already annotated "
                    f"({range_stats['targets_annotated']} of {range_stats['targets_total']}, "
                    f"{range_stats['refs_complete']} complete reference(s)). "
                    "Confirm re-annotation to overwrite existing annotations."
                ),
                **range_stats,
            }), 409
        # Partial: allow resume without force_reannotate (skip already annotated).
        # Overwrite requires force_reannotate=true after UI confirmation.

    cfg_path = BASE_DIR / "cascade" / "config.json"
    cfg = _load_cascade_config()
    try:
        ensure_ollama_ready(cfg, cfg_path, install_binary=False, pull_llm=True, quiet=True)
    except RuntimeError as e:
        return jsonify({
            'error': (
                f"Ollama / LLM unavailable: {e}. "
                "Try again in a few moments."
            )
        }), 503
    report = build_report(BASE_DIR, cfg)
    if not report.get("ready_for_run_model"):
        missing = report.get("missing_required") or []
        msg = "Incomplete configuration — Run Model unavailable."
        if missing:
            msg += " Missing: " + ", ".join(missing)
        return jsonify({"error": msg, "missing_required": missing}), 503

    backup_path = backup_json_file(file_path)
    _register_saved_session(file_path)
    targets_total = sum(
        len(data[idx].get("database") or [])
        for idx in range(start_index, end_index + 1)
    )

    started = start_batch(
        app,
        start_index=start_index,
        end_index=end_index,
        threshold=threshold,
        original_filename=original_filename,
        file_path=file_path,
        base_dir=BASE_DIR,
        targets_total=targets_total,
        force_reannotate=force_reannotate,
        backup_path=str(backup_path) if backup_path else None,
        app_module=sys.modules[__name__],
        cascade_mode=cascade_mode,
    )
    if not started:
        return jsonify({'error': 'Could not start batch.'}), 409

    return jsonify({
        'started': True,
        'message': 'Run Model started. Follow progress below.',
        'poll_url': '/api/auto_annotate/status',
        'backup': str(backup_path) if backup_path else None,
    }), 202

if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host=host, port=port, debug=debug, threaded=True)
