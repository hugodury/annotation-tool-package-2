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

    async function parseJsonResponse(resp) {
        const text = await resp.text();
        try {
            return JSON.parse(text);
        } catch (e) {
            if (resp.status === 404) {
                return { error: 'Server out of date — restart Flask (./start.sh)' };
            }
            return { error: 'Invalid server response' };
        }
    }

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

    window.loadStorageSettings = async function () {
        const input = document.getElementById('storageFolderInput');
        const hint = document.getElementById('storageFolderHint');
        const btnBrowse = document.getElementById('btnBrowseStorage');
        if (!input) return;
        const defaultHint = 'Sélecteur système natif (Linux, Windows, macOS).';
        const applyLockedState = function (locked, writable, dialogReady) {
            if (btnBrowse) btnBrowse.disabled = !!locked || dialogReady === false;
            if (btnChooseJson) btnChooseJson.disabled = dialogReady === false;
            if (!hint) return;
            let text = defaultHint;
            if (dialogReady === false) {
                text += ' Sélecteur indisponible sur cette machine — voir README (zenity/kdialog/yad, PowerShell, tkinter).';
            }
            if (locked) text += ' Verrouillé par ANNOTATION_DATA_DIR dans .env.';
            if (writable === false) text += ' Attention : dossier non accessible en écriture.';
            hint.textContent = text;
        };
        const btnChooseJson = document.getElementById('btnChooseJsonFile');
        try {
            const [storageResp, dialogResp] = await Promise.all([
                fetch('/api/storage'),
                fetch('/api/dialogs/capabilities'),
            ]);
            const data = await parseJsonResponse(storageResp);
            const dialogs = await parseJsonResponse(dialogResp);
            const dialogReady = dialogResp.ok && dialogs.ready !== false;
            if (storageResp.ok) {
                input.value = data.path || data.default_path || '';
                window.storageHomePath = data.home_path || data.path || '';
                applyLockedState(data.locked, data.writable, dialogReady);
                return;
            }
            const sessResp = await fetch('/api/sessions');
            const sess = await parseJsonResponse(sessResp);
            if (sessResp.ok && sess.storage_folder) {
                input.value = sess.storage_folder;
                applyLockedState(false, true, dialogReady);
                return;
            }
            applyLockedState(false, true, dialogReady);
        } catch (e) {
            applyLockedState(false, true, true);
        }
    };

    window.pickStorageFolder = async function () {
        const input = document.getElementById('storageFolderInput');
        const btnBrowse = document.getElementById('btnBrowseStorage');
        if (
            window.currentData
            && !confirm(
                'Change storage folder? The current session will be unloaded. Files in the previous folder are kept.'
            )
        ) {
            return;
        }
        if (btnBrowse) btnBrowse.disabled = true;
        showToast('Choose a folder in the system dialog…', 'info');
        try {
            const resp = await fetch('/api/storage/pick', { method: 'POST' });
            const data = await parseJsonResponse(resp);
            if (!resp.ok) {
                showToast(data.error || 'Could not open folder picker', 'danger');
                return;
            }
            if (data.cancelled) return;
            if (input) input.value = data.path || '';
            showToast(data.message || 'Storage folder updated', 'success');
            window.currentData = null;
            window.currentFilename = null;
            window.processedIds = [];
            updateActiveFileLabel(null);
            const nav = document.getElementById('navigationPanel');
            if (nav) nav.style.display = 'none';
            const single = document.getElementById('singleItemView');
            if (single) single.style.display = 'none';
            loadSessionsList();
        } catch (e) {
            showToast('Could not open folder picker', 'danger');
        } finally {
            loadStorageSettings();
        }
    };

    window.pickUploadJsonFile = async function () {
        const btn = document.getElementById('btnChooseJsonFile');
        if (
            window.currentData
            && !confirm(
                'Loading a new file will replace the current session. Continue?\n\nOr pick a session from the list on the left.'
            )
        ) {
            return;
        }
        if (btn) btn.disabled = true;
        showToast('Choose a JSON file in the system dialog…', 'info');
        try {
            const resp = await fetch('/api/upload/pick', { method: 'POST' });
            const data = await parseJsonResponse(resp);
            if (!resp.ok) {
                showToast(data.error || 'Could not open file picker', 'danger');
                return;
            }
            if (data.cancelled) return;
            window.currentData = data.data;
            window.processedIds = data.processed_ids || [];
            window.currentFilename = data.filename || null;
            window.currentFilePath = data.path || null;
            updateActiveFileLabel(data.filename);
            if (window.initializeUI) window.initializeUI();
            loadSessionsList();
            showToast('File loaded successfully.', 'success');
        } catch (e) {
            showToast('Could not open file picker', 'danger');
        } finally {
            if (btn) btn.disabled = false;
        }
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
        list.innerHTML = '<div class="session-row text-muted px-2">Loading sessions…</div>';
        try {
            const resp = await fetch('/api/sessions');
            const data = await parseJsonResponse(resp);
            if (!resp.ok) {
                list.innerHTML = '<div class="session-row text-muted px-2">Could not load sessions</div>';
                return;
            }
            window.storageFolderPath = data.storage_folder || '';
            const sessions = data.sessions || [];
            const labelEl = document.getElementById('savedSessionsLabel');
            if (labelEl) {
                labelEl.textContent = sessions.length
                    ? `Saved sessions (${sessions.length})`
                    : 'Saved sessions';
                labelEl.title = 'Files opened or used with Run Model in this app';
            }
            list.innerHTML = '';
            if (sel) sel.innerHTML = '<option value="">—</option>';
            if (!sessions.length) {
                list.innerHTML = '<div class="session-row text-muted px-2">No saved sessions yet — choose a JSON file or run Run Model</div>';
                if (data.current) updateActiveFileLabel(data.current);
                else if (!window.currentFilename) updateActiveFileLabel(null);
                return;
            }
            const home = (window.storageHomePath || '').replace(/\/$/, '');
            sessions.forEach((s) => {
                const storage = (data.storage_folder || '').replace(/\/$/, '');
                const sessionPath = s.path
                    || (s.parent ? `${s.parent.replace(/\/$/, '')}/${s.filename}` : '')
                    || (storage && s.filename ? `${storage}/${s.filename}` : s.filename || '');
                const isActive = sessionPath === data.current_path
                    || (s.filename === data.current && !data.current_path)
                    || s.filename === window.currentFilename;
                let folder = s.parent || '';
                if (home && folder.startsWith(home)) {
                    folder = '~' + folder.slice(home.length);
                }
                const row = document.createElement('div');
                row.className = 'session-row' + (isActive ? ' active' : '');
                row.innerHTML = `
                    <button type="button" class="session-load" title="${escapeHtml(sessionPath)}">
                        <span class="d-block">${escapeHtml(s.filename)} (${s.size_mb} MB)${isActive ? ' ★' : ''}</span>
                        <small class="text-muted">${escapeHtml(folder)}</small>
                    </button>
                    <button type="button" class="session-delete" title="Retirer de la liste ou supprimer du disque" aria-label="Supprimer">&times;</button>`;
                row.querySelector('.session-load').addEventListener('click', () => {
                    loadSessionFile(s.filename, sessionPath);
                });
                row.querySelector('.session-delete').addEventListener('click', (ev) => {
                    ev.stopPropagation();
                    removeSessionFromList(s.filename, sessionPath);
                });
                list.appendChild(row);
                if (sel) {
                    const opt = document.createElement('option');
                    opt.value = sessionPath;
                    opt.text = s.filename;
                    sel.appendChild(opt);
                }
            });
            if (data.current || window.currentFilename) {
                const active = data.current || window.currentFilename;
                if (sel && data.current_path) sel.value = data.current_path;
                updateActiveFileLabel(active);
            }
        } catch (e) {
            list.innerHTML = '<div class="session-row text-muted px-2">Could not load sessions</div>';
        }
    };

    window.promptSessionRemoval = function (filename, path) {
        return new Promise((resolve) => {
            const modalEl = document.getElementById('sessionRemoveModal');
            if (!modalEl || typeof bootstrap === 'undefined') {
                const choice = confirm(
                    `Session : "${filename}"\n\n` +
                    'OK = Retirer de la liste seulement (fichier conservé)\n' +
                    'Annuler = ne rien faire'
                );
                resolve(choice ? 'list' : 'cancel');
                return;
            }
            const nameEl = document.getElementById('sessionRemoveFilename');
            const pathEl = document.getElementById('sessionRemovePath');
            const listBtn = document.getElementById('sessionRemoveListBtn');
            const diskBtn = document.getElementById('sessionRemoveDiskBtn');
            if (nameEl) nameEl.textContent = filename;
            if (pathEl) pathEl.textContent = path || filename;
            const modal = bootstrap.Modal.getOrCreateInstance(modalEl);
            let settled = false;
            const finish = (value) => {
                if (settled) return;
                settled = true;
                listBtn?.removeEventListener('click', onList);
                diskBtn?.removeEventListener('click', onDisk);
                modalEl.removeEventListener('hidden.bs.modal', onHidden);
                resolve(value);
                modal.hide();
            };
            const onList = () => finish('list');
            const onDisk = () => finish('disk');
            const onHidden = () => finish('cancel');
            listBtn?.addEventListener('click', onList);
            diskBtn?.addEventListener('click', onDisk);
            modalEl.addEventListener('hidden.bs.modal', onHidden);
            modal.show();
        });
    };

    window.removeSessionFromList = async function (filename, path) {
        if (!filename) return;
        const mode = await window.promptSessionRemoval(filename, path || filename);
        if (mode === 'cancel') return;
        if (mode === 'disk') {
            const target = path || filename;
            if (!confirm(
                `Confirmer la suppression définitive ?\n\nLe fichier sera effacé du disque :\n${target}`
            )) {
                return;
            }
        }
        try {
            const resp = await fetch('/api/sessions/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, path: path || '', mode }),
            });
            const data = await resp.json();
            if (!resp.ok) return showToast(data.error || 'Action impossible', 'danger');
            if (window.currentFilename === filename) {
                window.currentData = null;
                window.currentFilename = null;
                window.processedIds = [];
                document.getElementById('navigationPanel').style.display = 'none';
                document.getElementById('singleItemView').style.display = 'none';
                updateActiveFileLabel(null);
            }
            showToast(data.message || 'Session mise à jour', 'success');
            loadSessionsList();
        } catch (e) {
            showToast('Erreur réseau', 'danger');
        }
    };

    window.loadSessionFile = async function (filename, path) {
        if (!filename) return;
        try {
            const resp = await fetch('/api/sessions/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, path: path || '' }),
            });
            const data = await parseJsonResponse(resp);
            if (!resp.ok) return showToast(data.error || 'Load failed', 'danger');
            window.currentData = data.data;
            window.processedIds = data.processed_ids;
            window.currentFilename = data.filename;
            window.currentFilePath = data.path || path || null;
            updateActiveFileLabel(data.filename);
            if (window.initializeUI) window.initializeUI();
            loadSessionsList();
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
        const btnCancel = document.getElementById('btnCancelRunModel');
        if (btnCancel) btnCancel.disabled = true;
        try {
            const resp = await fetch('/api/auto_annotate/cancel', { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok) {
                if (btnCancel) btnCancel.disabled = false;
                showToast(data.error || 'No batch running', 'warning');
                return;
            }
            if (msgEl) msgEl.textContent = 'Cancellation requested… stopping current target';
            showToast('Cancelling batch…', 'warning');
            if (typeof startRunModelPoll === 'function') startRunModelPoll(true);
        } catch (e) {
            if (btnCancel) btnCancel.disabled = false;
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

    window.clearRunLogs = async function () {
        if (!confirm('Clear Run Model logs for this server session?')) return;
        try {
            const resp = await fetch('/api/logs/clear', { method: 'POST' });
            const data = await parseJsonResponse(resp);
            if (!resp.ok) return showToast(data.error || 'Could not clear logs', 'danger');
            const pre = document.getElementById('runLogContent');
            if (pre) pre.textContent = '(no log yet)';
            showToast(data.message || 'Logs cleared', 'success');
        } catch (e) {
            showToast('Network error', 'danger');
        }
    };

    document.addEventListener('DOMContentLoaded', () => {
        const sessionSelect = document.getElementById('sessionSelect');
        if (sessionSelect) {
            sessionSelect.addEventListener('change', () => {
                const path = sessionSelect.value;
                if (!path) return;
                const filename = path.split(/[\\/]/).pop();
                loadSessionFile(filename, path);
            });
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
        const btnBrowseStorage = document.getElementById('btnBrowseStorage');
        if (btnBrowseStorage) btnBrowseStorage.addEventListener('click', pickStorageFolder);
        const btnChooseJson = document.getElementById('btnChooseJsonFile');
        if (btnChooseJson) btnChooseJson.addEventListener('click', pickUploadJsonFile);
        const btnClearLogs = document.getElementById('clearLogs');
        if (btnClearLogs) btnClearLogs.addEventListener('click', clearRunLogs);
        loadStorageSettings();
        loadSessionsList();
    });
})();
