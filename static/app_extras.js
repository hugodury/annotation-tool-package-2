/** UI extensions — toasts, sessions, filters, smart navigation */
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
        if (el) el.textContent = filename ? `Active file: ${filename}` : 'Active file: —';
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
        el.innerHTML = `<strong>Corpus:</strong> ${complete} complete, ${partial} partial, ${pending} pending — <strong>${pct}%</strong> done — <strong>${review}</strong> target(s) need validation`;
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
        const list = document.getElementById('sessionList');
        const sel = document.getElementById('sessionSelect');
        if (!list) return;
        try {
            const resp = await fetch('/api/sessions');
            const data = await resp.json();
            list.innerHTML = '';
            if (sel) sel.innerHTML = '<option value="">—</option>';
            const sessions = data.sessions || [];
            if (!sessions.length) {
                list.innerHTML = '<div class="session-row text-muted px-2">No saved sessions</div>';
                return;
            }
            sessions.forEach((s) => {
                const isActive = s.filename === data.current;
                const row = document.createElement('div');
                row.className = 'session-row' + (isActive ? ' active' : '');
                row.innerHTML = `
                    <button type="button" class="session-load" title="Load this session">
                        ${escapeHtml(s.filename)} (${s.size_mb} MB)${isActive ? ' ★' : ''}
                    </button>
                    <button type="button" class="session-delete" title="Delete this file" aria-label="Delete">&times;</button>`;
                row.querySelector('.session-load').addEventListener('click', () => {
                    loadSessionFile(s.filename);
                });
                row.querySelector('.session-delete').addEventListener('click', (ev) => {
                    ev.stopPropagation();
                    deleteSessionFile(s.filename);
                });
                list.appendChild(row);
                if (sel) {
                    const opt = document.createElement('option');
                    opt.value = s.filename;
                    opt.text = s.filename;
                    sel.appendChild(opt);
                }
            });
            if (data.current) {
                if (sel) sel.value = data.current;
                updateActiveFileLabel(data.current);
            }
        } catch (e) { /* ignore */ }
    };

    window.deleteSessionFile = async function (filename) {
        if (!filename) return;
        if (!confirm(`Delete session "${filename}"?\nThe JSON file will be removed from uploads/ (cannot be undone).`)) {
            return;
        }
        try {
            const resp = await fetch('/api/sessions/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename }),
            });
            const data = await resp.json();
            if (!resp.ok) return showToast(data.error || 'Delete failed', 'danger');
            if (window.currentFilename === filename) {
                window.currentData = null;
                window.currentFilename = null;
                document.getElementById('navigationPanel').style.display = 'none';
                document.getElementById('singleItemView').style.display = 'none';
                updateActiveFileLabel(null);
            }
            showToast(data.message || 'Session deleted', 'success');
            loadSessionsList();
        } catch (e) {
            showToast('Network error', 'danger');
        }
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
            if (!resp.ok) return showToast(data.error || 'Load failed', 'danger');
            window.currentData = data.data;
            window.processedIds = data.processed_ids;
            window.currentFilename = data.filename;
            updateActiveFileLabel(data.filename);
            if (window.initializeUI) window.initializeUI();
            showToast('Session loaded: ' + data.filename, 'success');
        } catch (e) {
            showToast('Network error', 'danger');
        }
    };

    window.resyncDatabase = async function () {
        try {
            const resp = await fetch('/api/resync', { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok) return showToast(data.error || 'Resync failed', 'danger');
            window.currentData = data.data;
            window.processedIds = data.processed_ids;
            if (window.initializeUI) window.initializeUI();
            showToast(data.message || 'Resynced', 'success');
        } catch (e) {
            showToast('Resync failed', 'danger');
        }
    };

    window.cancelRunModel = async function () {
        const msgEl = document.getElementById('runProgressMessage');
        try {
            const resp = await fetch('/api/auto_annotate/cancel', { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok) {
                showToast(data.error || 'No batch running', 'warning');
                return;
            }
            if (msgEl) msgEl.textContent = 'Cancellation requested…';
            showToast('Cancelling batch…', 'warning');
        } catch (e) {
            showToast('Could not cancel batch', 'danger');
        }
    };

    window.showRunLog = async function () {
        const panel = document.getElementById('runLogPanel');
        const pre = document.getElementById('runLogContent');
        if (!panel || !pre) return;
        try {
            const resp = await fetch('/api/logs/run_model/latest');
            const data = await resp.json();
            pre.textContent = data.content || '(no log yet)';
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        } catch (e) {
            showToast('Logs unavailable', 'danger');
        }
    };

    document.addEventListener('DOMContentLoaded', () => {
        const sessionSelect = document.getElementById('sessionSelect');
        if (sessionSelect) {
            sessionSelect.addEventListener('change', () => loadSessionFile(sessionSelect.value));
        }
        const refFilter = document.getElementById('refFilter');
        if (refFilter) refFilter.addEventListener('change', applyRefFilter);
        const btnReview = document.getElementById('btnNextReview');
        if (btnReview) {
            btnReview.addEventListener('click', () => {
                const idx = findNextReviewIndex(window.currentIndex);
                if (idx == null) return showToast('No reference needs review', 'warning');
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
