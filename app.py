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
load_dotenv(BASE_DIR / ".env")
sys.path.insert(0, str(BASE_DIR / "scripts"))
from annotation_store import backup_json_file, file_lock_for, parse_index_range  # noqa: E402
from ollama_service import (  # noqa: E402
    ensure_ollama_ready,
    ensure_ollama_ready_async,
    ensure_state,
)
from system_check import build_report  # noqa: E402
from run_model_job import (  # noqa: E402
    estimate_batch,
    find_resume_index,
    get_job_status,
    is_job_running,
    request_cancel_batch,
    start_batch,
)

# Lazy loaded models
sbert_model = None
cascade_engine = None


def _format_elapsed(seconds: float) -> str:
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
    elapsed = round(time.monotonic() - batch_start, 1)
    return {"elapsed_seconds": elapsed, "elapsed_label": _format_elapsed(elapsed)}

def get_cascade_engine():
    global cascade_engine
    if cascade_engine is None:
        from cascade.core import CascadeEngine
        cascade_engine = CascadeEngine()
    return cascade_engine


def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        from sentence_transformers import SentenceTransformer
        model_path = BASE_DIR / 'models' / 'fine_tuned_sbert'
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        sbert_model = SentenceTransformer(str(model_path), device=device)
    return sbert_model

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{(BASE_DIR / "instance" / "annotations.db").as_posix()}'
app.config['UPLOAD_FOLDER'] = str(BASE_DIR / 'uploads')
db = SQLAlchemy(app)

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
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


# Au démarrage : installer/démarrer Ollama et télécharger le LLM en arrière-plan.
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
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            lock = file_lock_for(upload_path)
            with lock:
                with open(upload_path, 'w', encoding='utf-8') as f:
                    f.write(raw)
            raw = raw.replace(': NaN', ': null')
            data = json.loads(raw)
            app.config['current_filename'] = safe_name
            # If the file is a dict, wrap it in a list for uniformity
            if isinstance(data, dict):
                data = [data]
            sync_file_annotations(safe_name, data, source='import')
            db.session.commit()
            data = enrich_data_with_status(safe_name, data)
            processed_ids = get_processed_ids_for_file(safe_name)
            return jsonify({
                'data': data,
                'processed_ids': list(processed_ids),
                'filename': safe_name,
            })
        except Exception as e:
            return jsonify({'error': f'Invalid JSON file: {str(e)}'}), 400
    return jsonify({'error': 'Invalid file type'}), 400

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
    original_filename = app.config.get('current_filename')
    if not original_filename:
        upload_dir = app.config['UPLOAD_FOLDER']
        files = [f for f in os.listdir(upload_dir) if f.endswith('.json')]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)), reverse=True)
            original_filename = files[0]
            app.config['current_filename'] = original_filename
            
    if not original_filename:
        return jsonify({'error': 'No file to download'}), 400
        
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    return send_file(file_path, as_attachment=True, download_name=f"annotated_{original_filename}")

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    req = request.json or {}
    news_id = str(req.get('news_id', ''))
    annotation = req.get('annotation')

    if not news_id:
        return jsonify({'error': 'news_id manquant.'}), 400
    if not isinstance(annotation, list):
        return jsonify({'error': 'annotation invalide (liste attendue).'}), 400
    
    # Get the original filename, recovering from server reload if necessary
    original_filename = app.config.get('current_filename')
    if not original_filename:
        upload_dir = app.config['UPLOAD_FOLDER']
        files = [f for f in os.listdir(upload_dir) if f.endswith('.json')]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)), reverse=True)
            original_filename = files[0]
            app.config['current_filename'] = original_filename
            
    if not original_filename:
        return jsonify({'error': 'Original file not found. Please re-upload your file.'}), 400
        
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'Original file not found'}), 400
        
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
                        f'Nombre de cibles incorrect : {len(annotation)} envoyees, '
                        f'{len(db_list)} attendues.'
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

def _resolve_current_filename() -> str | None:
    original_filename = app.config.get('current_filename')
    if not original_filename:
        upload_dir = app.config['UPLOAD_FOLDER']
        files = [f for f in os.listdir(upload_dir) if f.endswith('.json')]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)), reverse=True)
            original_filename = files[0]
            app.config['current_filename'] = original_filename
    return original_filename


def _load_upload_data(filename: str) -> tuple[list, Path]:
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data, Path(file_path)


@app.route('/api/sessions')
def api_sessions():
    upload_dir = Path(app.config['UPLOAD_FOLDER'])
    sessions = []
    for path in upload_dir.glob('*.json'):
        sessions.append({
            'filename': path.name,
            'mtime': path.stat().st_mtime,
            'size_mb': round(path.stat().st_size / (1024 * 1024), 2),
        })
    sessions.sort(key=lambda s: s['mtime'], reverse=True)
    return jsonify({
        'sessions': sessions,
        'current': app.config.get('current_filename'),
    })


@app.route('/api/sessions/load', methods=['POST'])
def api_load_session():
    req = request.json or {}
    filename = secure_filename(req.get('filename', ''))
    if not filename:
        return jsonify({'error': 'Nom de fichier invalide.'}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(file_path):
        return jsonify({'error': 'Fichier introuvable.'}), 404
    app.config['current_filename'] = filename
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
    })


@app.route('/api/resync', methods=['POST'])
def api_resync():
    filename = _resolve_current_filename()
    if not filename:
        return jsonify({'error': 'Aucun fichier charge.'}), 400
    data, _ = _load_upload_data(filename)
    sync_file_annotations(filename, data, source='import')
    db.session.commit()
    data = enrich_data_with_status(filename, data)
    processed_ids = get_processed_ids_for_file(filename)
    return jsonify({
        'message': 'Base resynchronisee depuis le JSON.',
        'data': data,
        'processed_ids': list(processed_ids),
        'filename': filename,
    })


@app.route('/api/logs/run_model/latest')
def api_latest_run_log():
    logs_dir = BASE_DIR / 'logs'
    if not logs_dir.is_dir():
        return jsonify({'filename': None, 'content': ''})
    files = sorted(logs_dir.glob('run_model_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return jsonify({'filename': None, 'content': ''})
    latest = files[0]
    content = latest.read_text(encoding='utf-8', errors='replace')
    if len(content) > 12000:
        content = content[-12000:]
    return jsonify({'filename': latest.name, 'content': content})


@app.route('/api/auto_annotate/cancel', methods=['POST'])
def api_auto_annotate_cancel():
    if request_cancel_batch():
        return jsonify({'cancelled': True, 'message': 'Annulation demandee…'})
    return jsonify({'error': 'Aucun Run Model en cours.'}), 409


@app.route('/api/auto_annotate/status')
def api_auto_annotate_status():
    return jsonify(get_job_status())


@app.route('/api/auto_annotate/estimate', methods=['POST'])
def api_auto_annotate_estimate():
    req = request.json or {}
    original_filename = _resolve_current_filename()
    if not original_filename:
        return jsonify({'error': 'Aucun fichier charge.'}), 400
    data, _ = _load_upload_data(original_filename)
    try:
        start_index, end_index = parse_index_range(req, len(data))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    cfg = _load_cascade_config()
    report = build_report(BASE_DIR, cfg)
    is_cpu = report.get('hardware', {}).get('gpu', {}).get('device') == 'cpu'
    est = estimate_batch(data, start_index, end_index, is_cpu=is_cpu, cfg=cfg)
    return jsonify(est)


@app.route('/api/auto_annotate/resume', methods=['GET'])
def api_auto_annotate_resume():
    original_filename = _resolve_current_filename()
    if not original_filename:
        return jsonify({'error': 'Aucun fichier charge.'}), 400
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
        return jsonify({'error': 'Un Run Model est deja en cours.'}), 409

    req = request.json or {}
    try:
        threshold = float(req.get('threshold', 0.95))
    except (TypeError, ValueError):
        return jsonify({'error': 'Seuil threshold invalide.'}), 400

    original_filename = _resolve_current_filename()
    if not original_filename:
        return jsonify({'error': 'Fichier introuvable. Re-uploadez votre JSON.'}), 400

    data, file_path = _load_upload_data(original_filename)
    try:
        start_index, end_index = parse_index_range(req, len(data))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    cfg_path = BASE_DIR / "cascade" / "config.json"
    cfg = _load_cascade_config()
    try:
        ensure_ollama_ready(cfg, cfg_path, install_binary=False, pull_llm=True, quiet=True)
    except RuntimeError as e:
        return jsonify({
            'error': (
                f"Ollama / LLM indisponible : {e}. "
                "Reessayez dans quelques instants."
            )
        }), 503
    report = build_report(BASE_DIR, cfg)
    if not report.get("ready_for_run_model"):
        missing = report.get("missing_required") or []
        msg = "Configuration incomplete — Run Model indisponible."
        if missing:
            msg += " Manque : " + ", ".join(missing)
        return jsonify({"error": msg, "missing_required": missing}), 503

    backup_path = backup_json_file(file_path)
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
        app_module=sys.modules[__name__],
    )
    if not started:
        return jsonify({'error': 'Impossible de demarrer le batch.'}), 409

    return jsonify({
        'started': True,
        'message': 'Run Model demarre. Suivez la progression ci-dessous.',
        'poll_url': '/api/auto_annotate/status',
        'backup': str(backup_path) if backup_path else None,
    }), 202

if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host=host, port=port, debug=debug)
