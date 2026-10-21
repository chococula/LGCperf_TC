// LGCperf result API (Google Apps Script web app bound to the results spreadsheet).
// Each TC gets its own sheet tab; every row has a stable Row_ID. Tabs starting with "_" are internal.
//
// GET  ?action=data                                     -> { ok, data: { <TC>: [rows...] } }
// POST { token, row }                                   -> append one result row (test PC)
// POST { action: 'checkKey', key }                      -> verify the admin key
// POST { action: 'exclude', key, tc, ids, excluded, reason } -> flag rows excluded / restored
// POST { action: 'update',  key, tc, changes: [{ id, values: { column: value } }] } -> edit cells
// POST { action: 'delete',  key, tc, ids }              -> move rows to the _Deleted tab
// POST { action: 'add',     key, tc, row }              -> append a row by hand (or undo a delete)

// Must match "token" in result_upload_config.json on the test PC.
const UPLOAD_TOKEN = 'CHANGE_ME';
// Needed for every change made from the dashboard or editor. Use a different value from UPLOAD_TOKEN.
const ADMIN_KEY = 'CHANGE_ME_TOO';

const ID_COLUMN = 'Row_ID';
const READ_ONLY_COLUMNS = [ID_COLUMN, 'Uploaded_At'];
const EXCLUSION_COLUMNS = ['Excluded', 'Exclude_Reason', 'Excluded_At'];
const DELETED_SHEET = '_Deleted';

function doGet(e) {
  if (e && e.parameter && e.parameter.action === 'data') {
    return jsonResponse({ ok: true, data: getResults() });
  }
  return ContentService.createTextOutput('LGCperf API. Open the dashboard on GitHub Pages.');
}

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents);
    switch (body.action) {
      case 'checkKey':
        return jsonResponse({ ok: checkAdminKey(body.key) });
      case 'exclude':
        return jsonResponse({ ok: true, row: setExcluded(body.tc, body.ids, body.excluded, body.reason, body.key) });
      case 'update':
        return jsonResponse({ ok: true, updated: updateRows(body.tc, body.changes, body.key) });
      case 'delete':
        return jsonResponse({ ok: true, deleted: deleteRows(body.tc, body.ids, body.key) });
      case 'add':
        return jsonResponse({ ok: true, id: addRow(body.tc, body.row, body.key) });
    }

    if (body.token !== UPLOAD_TOKEN) {
      return jsonResponse({ ok: false, error: 'invalid token' });
    }
    const row = body.row || {};
    const tc = String(row.TC || 'Unknown').replace(/[^\w.-]/g, '_');
    withLock(function () { appendRow(getOrCreateSheet(tc), row); });
    return jsonResponse({ ok: true });
  } catch (error) {
    return jsonResponse({ ok: false, error: error.message || String(error) });
  }
}

function getResults() {
  return withLock(function () {
    const result = {};
    SpreadsheetApp.getActiveSpreadsheet().getSheets().forEach(function (sheet) {
      if (sheet.getName().charAt(0) === '_' || sheet.getLastRow() < 2) return;
      ensureRowIds(sheet);
      const values = sheet.getDataRange().getDisplayValues();
      const header = values[0];
      result[sheet.getName()] = values.slice(1).map(function (line) {
        const record = {};
        header.forEach(function (key, column) { if (key) record[key] = line[column]; });
        return record;
      });
    });
    return result;
  });
}

function checkAdminKey(key) {
  if (key !== ADMIN_KEY) throw new Error('Admin key is incorrect.');
  return true;
}

// Soft delete: rows stay in the sheet, only flagged, so every exclusion can be restored.
function setExcluded(tc, ids, excluded, reason, key) {
  checkAdminKey(key);
  const values = {
    Excluded: excluded ? 'Y' : '',
    Exclude_Reason: excluded ? String(reason || '') : '',
    Excluded_At: excluded ? now() : '',
  };
  updateRows(tc, [].concat(ids).map(function (id) { return { id: id, values: values }; }), key, true);
  return values;
}

function updateRows(tc, changes, key, allowNewColumns) {
  checkAdminKey(key);
  return withLock(function () {
    const sheet = requireSheet(tc);
    let header = ensureRowIds(sheet);
    if (allowNewColumns) header = ensureColumns(sheet, EXCLUSION_COLUMNS);
    const rowsById = rowNumbersById(sheet, header);
    let updated = 0;
    changes.forEach(function (change) {
      const rowNumber = rowsById[change.id];
      if (!rowNumber) throw new Error('Row ' + change.id + ' no longer exists. Reload the page.');
      Object.keys(change.values).forEach(function (column) {
        if (READ_ONLY_COLUMNS.indexOf(column) !== -1) throw new Error(column + ' cannot be edited.');
        const index = header.indexOf(column);
        if (index === -1) throw new Error('Unknown column: ' + column);
        sheet.getRange(rowNumber, index + 1).setValue(change.values[column]);
      });
      updated++;
    });
    return updated;
  });
}

// Hard delete with a safety copy: rows are appended to the _Deleted tab before removal.
function deleteRows(tc, ids, key) {
  checkAdminKey(key);
  return withLock(function () {
    const sheet = requireSheet(tc);
    const header = ensureRowIds(sheet);
    const rowsById = rowNumbersById(sheet, header);
    const rowNumbers = ids.map(function (id) {
      if (!rowsById[id]) throw new Error('Row ' + id + ' no longer exists. Reload the page.');
      return rowsById[id];
    }).sort(function (a, b) { return b - a; });

    const backup = getOrCreateSheet(DELETED_SHEET);
    const deletedAt = now();
    rowNumbers.slice().reverse().forEach(function (rowNumber) {
      const values = sheet.getRange(rowNumber, 1, 1, header.length).getValues()[0];
      const record = { Deleted_At: deletedAt, Source_TC: tc };
      header.forEach(function (column, index) { record[column] = values[index]; });
      appendRecord(backup, record);
    });
    rowNumbers.forEach(function (rowNumber) { sheet.deleteRow(rowNumber); });
    return rowNumbers.length;
  });
}

function addRow(tc, row, key) {
  checkAdminKey(key);
  const name = String(tc || row.TC || '').replace(/[^\w.-]/g, '_');
  if (!name || name.charAt(0) === '_') throw new Error('Choose a TC for the new row.');
  return withLock(function () {
    const sheet = getOrCreateSheet(name);
    const header = ensureRowIds(sheet);
    const record = Object.assign({}, row);
    // Keep the given Row_ID only when restoring a deleted row whose ID is not in use.
    if (!record[ID_COLUMN] || rowNumbersById(sheet, header)[record[ID_COLUMN]]) record[ID_COLUMN] = Utilities.getUuid();
    if (!record.Uploaded_At) record.Uploaded_At = new Date();
    appendRecord(sheet, record);
    return record[ID_COLUMN];
  });
}

function requireSheet(tc) {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(tc);
  if (!sheet || tc.charAt(0) === '_') throw new Error('Sheet not found: ' + tc);
  return sheet;
}

function getOrCreateSheet(name) {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  return spreadsheet.getSheetByName(name) || spreadsheet.insertSheet(name);
}

function readHeader(sheet) {
  return sheet.getLastColumn() > 0
    ? sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0].filter(String)
    : [];
}

function ensureColumns(sheet, keys) {
  let header = readHeader(sheet);
  const missing = keys.filter(function (key) { return header.indexOf(key) === -1; });
  if (missing.length) {
    sheet.getRange(1, header.length + 1, 1, missing.length).setValues([missing]);
    header = header.concat(missing);
    sheet.setFrozenRows(1);
  }
  return header;
}

// Give every data row a Row_ID (rows added by hand or before IDs existed get one here).
function ensureRowIds(sheet) {
  const header = ensureColumns(sheet, [ID_COLUMN]);
  const lastRow = sheet.getLastRow();
  if (lastRow < 2) return header;
  const range = sheet.getRange(2, header.indexOf(ID_COLUMN) + 1, lastRow - 1, 1);
  const ids = range.getValues();
  let changed = false;
  ids.forEach(function (cell) {
    if (!cell[0]) { cell[0] = Utilities.getUuid(); changed = true; }
  });
  if (changed) range.setValues(ids);
  return header;
}

function rowNumbersById(sheet, header) {
  const result = {};
  const lastRow = sheet.getLastRow();
  if (lastRow < 2) return result;
  sheet.getRange(2, header.indexOf(ID_COLUMN) + 1, lastRow - 1, 1).getValues()
    .forEach(function (cell, index) { result[cell[0]] = index + 2; });
  return result;
}

function appendRow(sheet, row) {
  appendRecord(sheet, Object.assign({ Uploaded_At: new Date(), Row_ID: Utilities.getUuid() }, row));
}

function appendRecord(sheet, record) {
  const header = ensureColumns(sheet, Object.keys(record));
  sheet.appendRow(header.map(function (key) { return key in record ? record[key] : ''; }));
}

function now() {
  return Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm:ss');
}

function withLock(fn) {
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    return fn();
  } finally {
    lock.releaseLock();
  }
}

function jsonResponse(payload) {
  return ContentService.createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}
