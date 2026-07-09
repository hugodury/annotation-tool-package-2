from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
import json
import os
import sys
import time
from datetime import datetime
import math
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from ollama_service import (  # noqa: E402
    ensure_ollama_ready,
    ensure_ollama_ready_async,
    ensure_state,
)
from system_check import build_report  # noqa: E402

# Lazy loaded models
cross_encoder_model = None
sbert_model = None
cascade_engine = None

def get_cascade_engine():
    global cascade_engine
    if cascade_engine is None:
        from cascade.core import CascadeEngine
        cascade_engine = CascadeEngine()
    return cascade_engine


def get_cross_encoder():
    global cross_encoder_model
    if cross_encoder_model is None:
        from sentence_transformers.cross_encoder import CrossEncoder
        model_path = BASE_DIR / 'models' / 'fine_tuned_cross_encoder'
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        cross_encoder_model = CrossEncoder(str(model_path), device=device)
    return cross_encoder_model

def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        from sentence_transformers import SentenceTransformer
        model_path = BASE_DIR / 'models' / 'fine_tuned_sbert'
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        sbert_model = SentenceTransformer(str(model_path), device=device)
    return sbert_model

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

inverse_label_mapping = {
    0: "against",
    1: "not_related",
    2: "supporting",
    3: "undetermined"
}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///annotations.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)


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
        news_id = str(item.get('news_id', ''))
        content_status = reference_status_from_item(item)
        record = ReferenceAnnotation.query.filter_by(filename=filename, news_id=news_id).first()
        db_status = record.status if record else 'pending'
        if 'complete' in (content_status, db_status):
            final_status = 'complete'
        elif 'partial' in (content_status, db_status):
            final_status = 'partial'
        else:
            final_status = 'pending'
        item['annotation_status'] = final_status
        item['is_processed'] = final_status == 'complete'
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
    cfg_path = BASE_DIR / "cascade" / "config.json"
    return json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}


# Au démarrage : installer/démarrer Ollama et télécharger le LLM en arrière-plan.
ensure_ollama_ready_async(_load_cascade_config(), BASE_DIR / "cascade" / "config.json")

@app.route('/')
def index():
    return render_template('index.html')


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
            file.seek(0)
            raw = file.read().decode('utf-8')
            # Save the uploaded file to uploads directory
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            with open(upload_path, 'w', encoding='utf-8') as f:
                f.write(raw)
            # Replace NaN with null for valid JSON
            raw = raw.replace(': NaN', ': null')
            data = json.loads(raw)
            # Save the filename for later updates
            app.config['current_filename'] = file.filename
            # If the file is a dict, wrap it in a list for uniformity
            if isinstance(data, dict):
                data = [data]
            sync_file_annotations(file.filename, data, source='import')
            db.session.commit()
            data = enrich_data_with_status(file.filename, data)
            processed_ids = get_processed_ids_for_file(file.filename)
            return jsonify({'data': data, 'processed_ids': list(processed_ids), 'filename': file.filename})
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
    req = request.json
    news_id = str(req.get('news_id'))
    annotation = req.get('annotation')
    
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
            # Write similarity_annotation and related into each news in database
            db_list = item.get('database', [])
            if annotation and len(annotation) == len(db_list):
                for i, ann in enumerate(annotation):
                    db_list[i]['similarity_annotation'] = ann.get('similarity')
                    db_list[i]['related'] = ann.get('relation')
            updated_item = item
            break
    if updated_item is None:
        return jsonify({'error': 'News ID not found in file'}), 400
    # Save back to file
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

@app.route('/auto_annotate', methods=['POST'])
def auto_annotate():
    req = request.json
    start_index = int(req.get('start_index', 0))
    end_index = int(req.get('end_index', 0))
    threshold = float(req.get('threshold', 0.95))
    
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
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
        
    if start_index < 0 or end_index >= len(data) or start_index > end_index:
        return jsonify({'error': 'Invalid index range'}), 400

    cfg_path = BASE_DIR / "cascade" / "config.json"
    cfg = _load_cascade_config()
    try:
        ensure_ollama_ready(cfg, cfg_path, install_binary=False, pull_llm=True, quiet=True)
    except RuntimeError as e:
        return jsonify({
            'error': (
                f"Ollama / LLM indisponible : {e}. "
                "L'application tente de démarrer Ollama automatiquement — réessayez dans quelques instants."
            )
        }), 503
    report = build_report(BASE_DIR, cfg)
    if not report.get("ready_for_run_model"):
        missing = report.get("missing_required") or []
        msg = "Configuration incomplete — Run Model indisponible."
        if missing:
            msg += " Manque : " + ", ".join(missing)
        return jsonify({"error": msg, "missing_required": missing}), 503

    try:
        engine = get_cascade_engine()
        sbert = get_sbert_model()
    except Exception as e:
        return jsonify({'error': f'Failed to load models: {str(e)}'}), 500
    
    targets_annotated_count = 0
    total_targets_evaluated = 0
    references_fully_annotated_count = 0
    processed_ids = get_processed_ids_for_file(original_filename)
    
    routing_stats = {
        "deberta_auto": 0,
        "deberta_ambiguous": 0,
        "consensus": 0,
        "rejected": 0,
        "human": 0
    }

    run_cfg = cfg.get("run_model", {})
    no_annotation_timeout = int(run_cfg.get("no_annotation_timeout", 360))
    batch_start = time.monotonic()

    def _save_partial_and_respond(error_msg: str, status_code: int = 504):
        db.session.commit()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        enriched = enrich_data_with_status(original_filename, data)
        return jsonify({
            'error': error_msg,
            'message': (
                f"Run Model interrompu — {targets_annotated_count} cible(s) pre-remplie(s) "
                f"avant arret. Progression partielle sauvegardee."
            ),
            'annotated_count': references_fully_annotated_count,
            'data': enriched,
            'processed_ids': list(get_processed_ids_for_file(original_filename)),
            'routing_stats': routing_stats,
        }), status_code
    
    for idx in range(start_index, end_index + 1):
        item = data[idx]
        news_id = str(item.get('news_id'))
        
        # Skip already processed to avoid overwriting manual annotations
        if news_id in processed_ids:
            continue
            
        anchor_news = item.get('news', '')
        anchor_topic = item.get('topic', 'N/A') or 'N/A'
        anchor_date_raw = item.get('metadata', {}).get('date')
        anchor_date = str(anchor_date_raw)[:10] if anchor_date_raw else 'N/A'
        anchor_text = f"[Topic: {anchor_topic}] [Date: {anchor_date}] {anchor_news}"
        
        targets = item.get('database', [])
        
        if not targets:
            continue
            
        total_targets_evaluated += len(targets)
        
        all_above_threshold = True
        for i, target in enumerate(targets):
            if (
                targets_annotated_count == 0
                and time.monotonic() - batch_start > no_annotation_timeout
            ):
                return _save_partial_and_respond(
                    f"Aucune annotation automatique apres {no_annotation_timeout}s. "
                    "Verifiez Ollama, la RAM, ou reduisez la plage d'index."
                )

            target_news = target.get('news', '')
            target_topic = target.get('topic', 'N/A') or 'N/A'
            target_date_raw = target.get('metadata', {}).get('date')
            target_date = str(target_date_raw)[:10] if target_date_raw else 'N/A'
            target_text = f"[Topic: {target_topic}] [Date: {target_date}] {target_news}"
            
            # Execute Cascade routing
            out = engine.route(anchor_text, target_text, tau_auto=threshold)
            
            # Check for LLM service error
            if out.get("error") is not None:
                llm_tag = engine.llm_cfg.get("ollama", "LLM")
                return jsonify({
                    'error': (
                        f"Ollama service error: {out['error']}. "
                        f"Ensure Ollama is running and model '{llm_tag}' is pulled "
                        f"(run: python scripts/setup.py)."
                    )
                }), 503
                
            route_name = out["route"]
            target['cascade_route'] = route_name
            
            # If the route is an auto-annotation route:
            if route_name in {"deberta_auto", "deberta_ambiguous", "consensus"}:
                target['related'] = out["related"]
                target['model_confidence'] = round(out["deberta_conf"] if route_name != "consensus" else out["llm_conf"], 4)
                
                # Hybrid similarity: SBERT for DeBERTa, LLM score for LLM consensus
                if route_name == "consensus":
                    target['similarity_annotation'] = round(out["llm_sim"], 4)
                else:
                    # Compute SBERT similarity on the fly
                    emb_anchor = sbert.encode([anchor_text], show_progress_bar=False)[0]
                    emb_target = sbert.encode([target_text], show_progress_bar=False)[0]
                    sim = np.dot(emb_anchor, emb_target) / (np.linalg.norm(emb_anchor) * np.linalg.norm(emb_target))
                    target['similarity_annotation'] = round(max(0.0, min(1.0, float(sim))), 4)
                    
                targets_annotated_count += 1
                routing_stats[route_name] = routing_stats.get(route_name, 0) + 1
            else:
                # Routed to rejected or human -> needs manual review
                all_above_threshold = False
                
                # We do NOT pre-fill 'related' or 'similarity_annotation', so they stay blank.
                target['model_confidence'] = round(out["deberta_conf"], 4)
                if out["llm_pred"] is not None:
                    target['llm_pred'] = out["llm_pred"]
                    target['llm_confidence'] = round(out["llm_conf"], 4)
                    
                routing_stats[route_name] = routing_stats.get(route_name, 0) + 1
        
        if all_above_threshold:
            references_fully_annotated_count += 1

        sync_reference_to_db(original_filename, item, source='auto')
        processed_ids = get_processed_ids_for_file(original_filename)

    db.session.commit()
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    # Return updated full data
    data = enrich_data_with_status(original_filename, data)
    processed_ids = get_processed_ids_for_file(original_filename)
        
    msg = (
        f"Auto-annotation complete. Pre-filled {targets_annotated_count}/{total_targets_evaluated} targets. "
        f"Cascade Routing: {routing_stats['deberta_auto']} DeBERTa auto (>=95%), "
        f"{routing_stats['deberta_ambiguous']} DeBERTa ambiguous (against/not_related), "
        f"{routing_stats['consensus']} LLM consensus. "
        f"Flagged: {routing_stats['human']} for human review, {routing_stats['rejected']} rejected."
    )
        
    return jsonify({
        'message': msg,
        'annotated_count': references_fully_annotated_count,
        'data': data,
        'processed_ids': list(processed_ids)
    })

if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host=host, port=port, debug=debug)
