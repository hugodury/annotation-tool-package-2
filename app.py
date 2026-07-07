from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
import json
import os
import sys
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

class ProcessedNews(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    news_id = db.Column(db.String(50), unique=True, nullable=False)
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, news_id):
        self.news_id = news_id

with app.app_context():
    db.create_all()


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
            # Get processed ids
            processed_ids = set(row.news_id for row in ProcessedNews.query.all())
            # Mark processed state in the data
            for item in data:
                item['is_processed'] = str(item.get('news_id')) in processed_ids
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
        
    processed_ids = set(row.news_id for row in ProcessedNews.query.all())
    for item in data:
        item['is_processed'] = str(item.get('news_id')) in processed_ids
        
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
    updated = False
    for item in data:
        if str(item.get('news_id')) == news_id:
            # Write similarity_annotation and related into each news in database
            db_list = item.get('database', [])
            if annotation and len(annotation) == len(db_list):
                for i, ann in enumerate(annotation):
                    db_list[i]['similarity_annotation'] = ann.get('similarity')
                    db_list[i]['related'] = ann.get('relation')
            updated = True
            break
    if not updated:
        return jsonify({'error': 'News ID not found in file'}), 400
    # Save back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    # Mark as processed in DB
    if not ProcessedNews.query.filter_by(news_id=news_id).first():
        db.session.add(ProcessedNews(news_id=news_id))
        db.session.commit()
    return jsonify({'message': 'Annotation saved successfully'})

@app.route('/clear_database', methods=['POST'])
def clear_database():
    try:
        ProcessedNews.query.delete()
        db.session.commit()
        return jsonify({'message': 'Database cleared successfully'})
    except Exception as e:
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
    # Run Model toujours autorisé — les avertissements sont informatifs seulement.
        
    try:
        engine = get_cascade_engine()
        sbert = get_sbert_model()
    except Exception as e:
        return jsonify({'error': f'Failed to load models: {str(e)}'}), 500
    
    targets_annotated_count = 0
    total_targets_evaluated = 0
    references_fully_annotated_count = 0
    processed_ids = set(row.news_id for row in ProcessedNews.query.all())
    
    routing_stats = {
        "deberta_auto": 0,
        "deberta_ambiguous": 0,
        "consensus": 0,
        "rejected": 0,
        "human": 0
    }
    
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
            db.session.add(ProcessedNews(news_id=news_id))
            references_fully_annotated_count += 1
            processed_ids.add(news_id)
            
    db.session.commit()
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    # Return updated full data
    for item in data:
        item['is_processed'] = str(item.get('news_id')) in processed_ids
        
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
