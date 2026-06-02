#!/usr/bin/env node
/**
 * reembed.mjs — smart-connections 임베딩 인덱스 incremental 재임베딩 (standalone).
 * 2026-06-02: MCP는 startup에 무거운 모델을 못 올림(init timeout) → on-demand rebuild만 존재 →
 * 새 노트가 dense 인덱스에 안 들어가 stale. 이 스크립트를 launchd(일일)로 돌려 자동 freshness 유지.
 * buildIndex는 incremental(변경/신규만 임베딩) + saveIndex 자동. SMART_VAULT_PATH 필요.
 * calibration: ~/.claude/logs/calibration/smartconn_reembed.log (processed/unchanged 추이).
 */
import { SmartConnectionsLoader } from './dist/smart-connections-loader.js';
import { GteEmbedder } from './dist/gte-embedder.js';
import fs from 'fs';
import path from 'path';
import os from 'os';

const VAULT = process.env.SMART_VAULT_PATH;
if (!VAULT) { console.error('SMART_VAULT_PATH 미설정'); process.exit(1); }

const loader = new SmartConnectionsLoader(VAULT);
await loader.initialize();
const emb = new GteEmbedder(VAULT);
await emb.initialize();

const before = emb.getStats()?.entries ?? 0;
const notePaths = Array.from(loader.getSources().keys());
console.error(`reembed: ${notePaths.length} sources, index entries before=${before}`);

const stats = await emb.buildIndex(
  notePaths,
  (p) => loader.readNoteContent(p),
  (c, t) => { if (c % 50 === 0 || c === t) console.error(`  embedding ${c}/${t}`); }
);
const after = emb.getStats()?.entries ?? 0;

const ts = new Date().toISOString();
const line = `${ts} processed=${stats.notes_processed} unchanged=${stats.notes_unchanged} entries=${before}->${after}`;
const logDir = path.join(os.homedir(), '.claude', 'logs', 'calibration');
fs.mkdirSync(logDir, { recursive: true });
fs.appendFileSync(path.join(logDir, 'smartconn_reembed.log'), line + '\n');
console.log('REEMBED_DONE ' + line);
