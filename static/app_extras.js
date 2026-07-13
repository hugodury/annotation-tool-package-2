/** Extensions UI — toasts, sessions, filtres, navigation intelligente */
(function () {
    'use strict';

    window.escapeHtml = function (text) {
        if (text == null) return '';
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    };

    window.showToast = function (message, type) {
        const host = document.getElementById('toastHost');
        if (!host) return;
        const el = document.createElement('div');
        el.className = 'toast align-items-center text-bg-' + (type || 'primary') + ' border-0 show mb-2';
        el.setAttribute('role', 'alert');
        el.innerHTML = `<div class="d-flex"><div class="toast-body">${escapeHtml(message)}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
        host.appendChild(el);
        setTimeout(() => el.remove(), 5000);
    };

    window.parseIndexInputs = function () {
        const start = parseInt(document.getElementById('autoStart')?.value, 10);
        const end = parseInt(document.getElementById('autoEnd')?.value, 10);
        if (Number.isNaN(start) || Number.isNaN(end)) return null;
        return { start, end };
    };

    window.updateActiveFileLabel = function (filename) {
        const el = document.getElementById('activeFileLabel');
        if (el) el.textContent = filename ? `Fichier actif : ${filename}` : 'Fichier actif : —';
    };

    window.updateCorpusStats = function () {
        const el = document.getElementById('corpusStats');
        if (!el || !window.currentData) {
            if (el) el.style.display = 'none';
            return;
        }
        const data = window.currentData;
        let complete = 0, partial = 0, pending = 0, review = 0;
        data.forEach((item) => {
            const st = item.annotation_status || 'pending';
            if (st === 'complete') complete++;
            else if (st === 'partial') partial++;
            else pending++;
            (item.database || []).forEach((t) => {
                if (t.cascade_route === 'human' || t.cascade_route === 'rejected') review++;
            });
        });
        const pct = data.length ? Math.round((complete / data.length) * 100) : 0;
        el.style.display = 'block';
        el.innerHTML = `<strong>Corpus :</strong> ${complete} complete(s), ${partial} partielle(s), ${pending} en attente — <strong>${pct}%</strong> termine — <strong>${review}</strong> cible(s) a revoir`;
    };

    window.findNextIncompleteIndex = function (from) {
        if (!window.currentData) return null;
        for (let i = from + 1; i < window.currentData.length; i++) {
            if (window.currentData[i].annotation_status !== 'complete') return i;
        }
        return null;
    };

    window.findNextReviewIndex = function (from) {
        if (!window.currentData) return null;
        const start = from == null ? 0 : from + 1;
        for (let i = start; i < window.currentData.length; i++) {
            const targets = window.currentData[i].database || [];
            for (const t of targets) {
                if (t.cascade_route === 'human' || t.cascade_route === 'rejected') return i;
                if (t.cascade_route && !t.related) return i;
            }
        }
        return null;
    };

    window.applyRefFilter = function () {
        const filter = document.getElementById('refFilter')?.value || 'all';
        const select = document.getElementById('referenceSelect');
        if (!select || !window.currentData) return;
        const prev = window.currentIndex;
        select.innerHTML = '';
        window.currentData.forEach((item, idx) => {
            const st = item.annotation_status || 'pending';
            let show = true;
            if (filter === 'complete') show = st === 'complete';
            else if (filter === 'partial') show = st === 'partial';
            else if (filter === 'pending') show = st === 'pending';
            else if (filter === 'review') {
                show = (item.database || []).some(
                    (t) => t.cascade_route === 'human' || t.cascade_route === 'rejected'
                );
            }
            if (!show) return;
            const icon = window.statusIcon ? window.statusIcon(item) : '';
            const opt = document.createElement('option');
            opt.value = idx;
            opt.text = `[${idx}] ${icon} ${(item.news || '').substring(0, 60)}...`;
            select.appendChild(opt);
        });
        if (select.options.length) {
            const hasPrev = Array.from(select.options).some((o) => parseInt(o.value, 10) === prev);
            select.value = hasPrev ? String(prev) : select.options[0].value;
            window.currentIndex = parseInt(select.value, 10);
            if (window.displayCurrentItem) window.displayCurrentItem();
        }
    };

    window.loadSessionsList = async function () {
        const sel = document.getElementById('sessionSelect');
        if (!sel) return;
        try {
            const resp = await fetch('/api/sessions');
            const data = await resp.json();
            sel.innerHTML = '<option value="">— Choisir une session —</option>';
            (data.sessions || []).forEach((s) => {
                const opt = document.createElement('option');
                opt.value = s.filename;
                const mark = s.filename === data.current ? ' ★' : '';
                opt.text = `${s.filename} (${s.size_mb} Mo)${mark}`;
                sel.appendChild(opt);
            });
            if (data.current) {
                sel.value = data.current;
                updateActiveFileLabel(data.current);
            }
        } catch (e) { /* ignore */ }
    };

    window.loadSessionFile = async function (filename) {
        if (!filename) return;
        try {
            const resp = await fetch('/api/sessions/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename }),
            });
            const data = await resp.json();
            if (!resp.ok) return showToast(data.error || 'Erreur chargement', 'danger');
            window.currentData = data.data;
            window.processedIds = data.processed_ids;
            window.currentFilename = data.filename;
            updateActiveFileLabel(data.filename);
            if (window.initializeUI) window.initializeUI();
            showToast('Session chargee : ' + data.filename, 'success');
        } catch (e) {
            showToast('Erreur reseau', 'danger');
        }
    };

    window.resyncDatabase = async function () {
        try {
            const resp = await fetch('/api/resync', { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok) return showToast(data.error || 'Erreur resync', 'danger');
            window.currentData = data.data;
            window.processedIds = data.processed_ids;
            if (window.initializeUI) window.initializeUI();
            showToast(data.message || 'Resynchronise', 'success');
        } catch (e) {
            showToast('Erreur resync', 'danger');
        }
    };

    window.cancelRunModel = async function () {
        const msgEl = document.getElementById('runProgressMessage');
        try {
            const resp = await fetch('/api/auto_annotate/cancel', { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok) {
                showToast(data.error || 'Aucun batch en cours', 'warning');
                return;
            }
            if (msgEl) msgEl.textContent = 'Annulation demandee…';
            showToast('Annulation en cours…', 'warning');
        } catch (e) {
            showToast('Impossible d annuler', 'danger');
        }
    };

    window.showRunLog = async function () {
        const panel = document.getElementById('runLogPanel');
        const pre = document.getElementById('runLogContent');
        if (!panel || !pre) return;
        try {
            const resp = await fetch('/api/logs/run_model/latest');
            const data = await resp.json();
            pre.textContent = data.content || '(aucun log)';
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        } catch (e) {
            showToast('Logs indisponibles', 'danger');
        }
    };

    document.addEventListener('DOMContentLoaded', () => {
        loadSessionsList();

        const sessionSelect = document.getElementById('sessionSelect');
        if (sessionSelect) {
            sessionSelect.addEventListener('change', () => loadSessionFile(sessionSelect.value));
        }
        const resyncBtn = document.getElementById('resyncDb');
        if (resyncBtn) resyncBtn.addEventListener('click', resyncDatabase);
        const refFilter = document.getElementById('refFilter');
        if (refFilter) refFilter.addEventListener('change', applyRefFilter);
        const btnReview = document.getElementById('btnNextReview');
        if (btnReview) {
            btnReview.addEventListener('click', () => {
                const idx = findNextReviewIndex(window.currentIndex ?? -1);
                if (idx == null) return showToast('Aucune reference a revoir', 'warning');
                window.currentIndex = idx;
                if (window.displayCurrentItem) window.displayCurrentItem();
            });
        }
        const btnCancel = document.getElementById('btnCancelRunModel');
        if (btnCancel) btnCancel.addEventListener('click', cancelRunModel);
        const btnLog = document.getElementById('btnShowLogs');
        if (btnLog) btnLog.addEventListener('click', showRunLog);
    });
})();
