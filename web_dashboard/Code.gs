// LGCperf result API + dashboard (Google Apps Script web app bound to the results spreadsheet).
// Each TC gets its own sheet tab; rows are appended by the test PC and read by the dashboard.
//
// GET  ?action=data                         -> { ok, data: { <TC>: [rows...] } }
// GET  (no action)                          -> dashboard page (Index.html)
// POST { token, row }                       -> append one result row (test PC)
// POST { action: 'checkKey', key }          -> verify the admin key (dashboard)
// POST { action: 'exclude', key, tc, row, excluded, reason } -> mark a row excluded / restored

// Must match "token" in result_upload_config.json on the test PC.
const UPLOAD_TOKEN = 'CHANGE_ME';
// Needed to exclude or restore rows from the dashboard. Use a different value from UPLOAD_TOKEN.
const ADMIN_KEY = 'CHANGE_ME_TOO';

const EXCLUSION_COLUMNS = ['Excluded', 'Exclude_Reason', 'Excluded_At'];

function doGet(e) {
  if (e && e.parameter && e.parameter.action === 'data') {
    return jsonResponse({ ok: true, data: getResults() });
  }
  return HtmlService.createHtmlOutputFromFile('Index')
    .setTitle('LGCperf Results')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents);
    if (body.action === 'checkKey') {
      return jsonResponse({ ok: checkAdminKey(body.key) });
    }
    if (body.action === 'exclude') {
      return jsonResponse({ ok: true, row: setExcluded(body.tc, body.row, body.excluded, body.reason, body.key) });
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

// Also callable from Index.html via google.script.run when the page is served by this script.
function getResults() {
  const result = {};
  SpreadsheetApp.getActiveSpreadsheet().getSheets().forEach(function (sheet) {
    const values = sheet.getDataRange().getDisplayValues();
    if (values.length < 2) return;
    const header = values[0];
    result[sheet.getName()] = values.slice(1).map(function (line, index) {
      const record = { _row: index + 2 };
      header.forEach(function (key, column) { record[key] = line[column]; });
      return record;
    });
  });
  return result;
}

function checkAdminKey(key) {
  if (key !== ADMIN_KEY) throw new Error('Admin key is incorrect.');
  return true;
}

// Soft delete: rows are never removed, only flagged, so every exclusion can be restored.
function setExcluded(sheetName, rowNumber, excluded, reason, key) {
  checkAdminKey(key);
  return withLock(function () {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(sheetName);
    if (!sheet || rowNumber < 2 || rowNumber > sheet.getLastRow()) throw new Error('Row not found. Reload the page.');
    const header = ensureColumns(sheet, EXCLUSION_COLUMNS);
    const values = {
      Excluded: excluded ? 'Y' : '',
      Exclude_Reason: excluded ? String(reason || '') : '',
      Excluded_At: excluded ? Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm:ss') : '',
    };
    EXCLUSION_COLUMNS.forEach(function (name) {
      sheet.getRange(rowNumber, header.indexOf(name) + 1).setValue(values[name]);
    });
    return values;
  });
}

function getOrCreateSheet(name) {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  return spreadsheet.getSheetByName(name) || spreadsheet.insertSheet(name);
}

function ensureColumns(sheet, keys) {
  let header = sheet.getLastColumn() > 0
    ? sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0].filter(String)
    : [];
  const missing = keys.filter(function (key) { return header.indexOf(key) === -1; });
  if (missing.length) {
    sheet.getRange(1, header.length + 1, 1, missing.length).setValues([missing]);
    header = header.concat(missing);
    sheet.setFrozenRows(1);
  }
  return header;
}

function appendRow(sheet, row) {
  const header = ensureColumns(sheet, ['Uploaded_At'].concat(Object.keys(row)));
  const record = Object.assign({ Uploaded_At: new Date() }, row);
  sheet.appendRow(header.map(function (key) { return key in record ? record[key] : ''; }));
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
