import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const config = JSON.parse(fs.readFileSync(path.join(root, 'config/training_protocol_v1.json'), 'utf8'));
const version = config.version;
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const rank = (scope, value) => hash(`${config.seed}|${scope}|${value}`);
const invariant = (condition, message) => assert.ok(condition, message);

function parseCsv(text) {
  text = text.replace(/^\uFEFF/, '');
  const records = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(field);
      if (row.some(value => value !== '')) records.push(row);
      row = []; field = '';
    } else field += ch;
  }
  invariant(!quoted, 'Unterminated CSV quote');
  if (field !== '' || row.length) { row.push(field); records.push(row); }
  const headers = records.shift();
  invariant(headers && new Set(headers).size === headers.length, 'Invalid CSV header');
  return records.map(values => {
    invariant(values.length === headers.length, 'CSV column count mismatch');
    return Object.fromEntries(headers.map((key, i) => [key, values[i]]));
  });
}

function csvText(rows, headers = Object.keys(rows[0])) {
  const escape = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
  return [headers, ...rows.map(row => headers.map(key => row[key]))]
    .map(row => row.map(escape).join(',')).join('\r\n') + '\r\n';
}

const readCsv = relative => parseCsv(fs.readFileSync(path.join(root, relative), 'utf8'));
function groupBy(rows, getKey) {
  const groups = new Map();
  for (const row of rows) {
    const key = getKey(row);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return groups;
}

const manifestPath = 'data/metadata/manifest.csv';
const escPath = 'data/metadata/esc50_manifest.csv';
const manifest = readCsv(manifestPath);
const esc = readCsv(escPath);
const allIds = [...manifest, ...esc].map(row => row.sample_id);
invariant(new Set(allIds).size === allIds.length, 'Duplicate sample ID');
for (const row of [...manifest, ...esc]) {
  invariant(row.label in config.labels, `Unknown label: ${row.sample_id}`);
  invariant(fs.existsSync(path.join(root, row.relative_path)), `Missing audio: ${row.sample_id}`);
  row.label_id = String(config.labels[row.label]);
}

const knownDuplicates = manifest.filter(row => row.duplicate_of);
for (const row of knownDuplicates) {
  const original = manifest.find(candidate => candidate.sample_id === row.duplicate_of);
  invariant(original && !original.duplicate_of, 'Invalid duplicate reference');
  const digest = r => hash(fs.readFileSync(path.join(root, r.relative_path)));
  invariant(digest(row) === digest(original), 'Marked duplicate is not byte-identical');
  invariant(row.split_group_id === original.split_group_id, 'Duplicate group mismatch');
}

for (const row of manifest) {
  if (row.target_domain !== 'ko_target') continue;
  row.label_review_status = config.review.domestic_file_labels;
  row.label_review_date = config.confirmation_date;
  row.source_group_review_status = config.review.domestic_sources;
  row.source_group_review_date = config.confirmation_date;
  row.notes = row.notes
    .replace('initial class from filename, audio-content/overlap review pending', 'file-class label verified by user 2026-11-08')
    .replace('merge split_group_id if this clip shares a source video with another', 'distinct sources confirmed by user 2026-11-08, except documented exact duplicates')
    .replace('initial label and Korean target domain supplied by user; listening/overlap review pending', 'file-class label verified by user 2026-11-08; distinct sources confirmed by user except documented exact duplicates');
}

const active = manifest.filter(row => !row.duplicate_of);
const domestic = active.filter(row => row.target_domain === 'ko_target');
invariant(new Set(domestic.map(row => row.split_group_id)).size === domestic.length,
  'Protocol expects one canonical domestic recording per independently confirmed source');
function stratum(row) {
  if (row.label === 'doorbell') return row.source_dataset;
  if (row.label === 'fire_alarm') return Number(row.duration_s) <= 12 ? 'short_le_12s' : 'long_gt_12s';
  return 'all';
}

const inputIdentity = hash(JSON.stringify([...active, ...esc].map(row => [
  row.sample_id, row.label, row.target_domain, row.source_dataset, row.source_recording_id,
  row.duplicate_of || ''
]).sort((a, b) => a[0].localeCompare(b[0]))));
const outputDir = path.join(root, 'data/splits', version);
const lockPath = path.join(outputDir, 'holdout_lock.json');
let lock;
if (fs.existsSync(lockPath)) {
  lock = JSON.parse(fs.readFileSync(lockPath, 'utf8'));
  invariant(lock.input_identity === inputIdentity, 'Dataset inventory changed: retain test IDs and create a reviewed new protocol version');
  invariant(lock.config_hash === hash(JSON.stringify(config)), 'Protocol config changed: use a new version');
} else {
  const assignments = {};
  for (const label of Object.keys(config.labels)) {
    const rows = domestic.filter(row => row.label === label);
    const groups = [...groupBy(rows, stratum)].map(([name, members]) => ({
      name, members, quota: Math.floor(members.length * config.split.domestic_test_groups[label] / rows.length),
      remainder: members.length * config.split.domestic_test_groups[label] / rows.length % 1
    }));
    let remaining = config.split.domestic_test_groups[label] - groups.reduce((sum, group) => sum + group.quota, 0);
    for (const group of [...groups].sort((a, b) => b.remainder - a.remainder || a.name.localeCompare(b.name))) {
      if (remaining-- > 0) group.quota++;
    }
    const development = [];
    for (const group of groups) {
      const ranked = [...group.members].sort((a, b) => rank(`holdout:${label}`, a.split_group_id).localeCompare(rank(`holdout:${label}`, b.split_group_id)));
      ranked.slice(0, group.quota).forEach(row => { assignments[row.sample_id] = {test: true, cv_fold: null}; });
      development.push(...ranked.slice(group.quota));
    }
    const foldCounts = Array(config.split.development_folds).fill(0);
    const strata = [...groupBy(development, stratum)].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
    for (const [name, members] of strata) {
      const stratumCounts = Array(config.split.development_folds).fill(0);
      members.sort((a, b) => rank(`cv:${label}:${name}`, a.split_group_id).localeCompare(rank(`cv:${label}:${name}`, b.split_group_id)));
      for (const row of members) {
        const fold = foldCounts.map((_, i) => i).sort((a, b) => foldCounts[a] - foldCounts[b] || stratumCounts[a] - stratumCounts[b] || a - b)[0];
        assignments[row.sample_id] = {test: false, cv_fold: fold};
        foldCounts[fold]++; stratumCounts[fold]++;
      }
    }
  }
  lock = {version, seed: config.seed, config_hash: hash(JSON.stringify(config)), input_identity: inputIdentity,
    rule: 'metadata-stratified deterministic SHA256 ranking; no model scores or audio inspection used', assignments};
}

const escRepairs = [];
const priority = {train: 0, validation: 1, test: 2};
const confusers = new Set(config.sampling.esc50_confuser_categories);
for (const [source, rows] of groupBy(esc, row => row.source_recording_id)) {
  const role = [...rows].map(row => row.upstream_split || row.split).sort((a, b) => priority[b] - priority[a])[0];
  for (const row of rows) {
    row.upstream_split ||= row.split;
    row.split = role;
    row.split_group_id = `ESC50-source-${source}`;
    row.split_adjustment_reason = row.upstream_split === role ? '' : 'source_group_shared_with_more_restrictive_partition';
    const category = row.notes.match(/Original category=([^;]+)/)?.[1];
    invariant(category, `Missing original category: ${row.sample_id}`);
    row.confuser_tags = confusers.has(category) ? category : '';
    row.split_version = version;
    if (row.upstream_split !== role) escRepairs.push({sample_id: row.sample_id, source_recording_id: source, from: row.upstream_split, to: role});
  }
}

function splitRow(row, fold, refit = false) {
  const assignment = lock.assignments[row.sample_id];
  let split = row.split;
  if (row.source_dataset !== 'ESC-50') {
    split = !assignment ? 'train' : assignment.test ? 'test' : !refit && assignment.cv_fold === fold ? 'validation' : 'train';
  }
  const isDomestic = row.target_domain === 'ko_target';
  const pool = row.label !== 'other' ? (isDomestic ? 'domestic_target' : 'auxiliary_target')
    : isDomestic ? 'domestic_doorlock' : row.confuser_tags ? 'overseas_confuser' : 'overseas_general';
  return {sample_id: row.sample_id, relative_path: row.relative_path, label: row.label, label_id: String(config.labels[row.label]),
    target_domain: row.target_domain, source_dataset: row.source_dataset, source_recording_id: row.source_recording_id,
    split_group_id: row.split_group_id, split, split_version: version,
    cv_fold: assignment && !assignment.test ? String(assignment.cv_fold) : '', sampler_pool: pool};
}

function validateSplit(rows) {
  invariant(rows.length === active.length + esc.length, 'Split row count mismatch');
  invariant(new Set(rows.map(row => row.sample_id)).size === rows.length, 'Repeated split sample');
  for (const members of groupBy(rows, row => row.split_group_id).values()) {
    invariant(new Set(members.map(row => row.split)).size === 1, 'Source group crosses partitions');
  }
  for (const members of groupBy(rows, row => `${row.source_dataset}|${row.source_recording_id}`).values()) {
    invariant(new Set(members.map(row => row.split)).size === 1, 'Original source crosses partitions');
  }
  for (const label of Object.keys(config.labels)) {
    invariant(rows.filter(row => row.target_domain === 'ko_target' && row.label === label && row.split === 'test').length === config.split.domestic_test_groups[label], 'Domestic test quota mismatch');
  }
}

function samplingGroups(rows) {
  const training = rows.filter(row => row.split === 'train');
  const groups = [...groupBy(training, row => row.split_group_id)].map(([id, members]) => {
    invariant(new Set(members.map(row => row.label)).size === 1, 'Source group spans class labels');
    let pool = members[0].sampler_pool;
    if (members.some(row => row.sampler_pool === 'overseas_confuser')) pool = 'overseas_confuser';
    return {split_group_id: id, label: members[0].label, label_id: members[0].label_id, sampler_pool: pool,
      file_count: members.length, sample_ids: members.map(row => row.sample_id).sort().join(';'), group_probability: 0};
  });
  for (const label of Object.keys(config.labels)) {
    const members = groups.filter(group => group.label === label);
    const classProbability = config.sampling.class_probabilities[label];
    if (label !== 'other') {
      const weight = group => group.sampler_pool === 'domestic_target' ? config.sampling.target_class_domestic_group_weight : config.sampling.target_class_auxiliary_group_weight;
      const total = members.reduce((sum, group) => sum + weight(group), 0);
      for (const group of members) group.group_probability = classProbability * weight(group) / total;
    } else {
      for (const [pool, fraction] of Object.entries(config.sampling.other_pool_probabilities)) {
        const poolGroups = members.filter(group => group.sampler_pool === pool);
        invariant(poolGroups.length > 0, `Empty sampler pool: ${pool}`);
        for (const group of poolGroups) group.group_probability = classProbability * fraction / poolGroups.length;
      }
    }
    invariant(Math.abs(members.reduce((sum, group) => sum + group.group_probability, 0) - classProbability) < 1e-10, 'Class sampling probability mismatch');
  }
  invariant(Math.abs(groups.reduce((sum, group) => sum + group.group_probability, 0) - 1) < 1e-10, 'Sampler does not sum to one');
  return groups;
}

const folds = Array.from({length: config.split.development_folds}, (_, fold) => {
  const rows = [...active, ...esc].map(row => splitRow(row, fold));
  validateSplit(rows);
  return {rows, sampler: samplingGroups(rows)};
});
const refit = [...active, ...esc].map(row => splitRow(row, 0, true));
validateSplit(refit);
const defaultRows = new Map(folds[config.split.default_fold].rows.map(row => [row.sample_id, row]));
for (const row of manifest) {
  const selected = defaultRows.get(row.duplicate_of || row.sample_id);
  row.split = selected.split;
  row.cv_fold = selected.cv_fold;
  row.split_version = version;
}

const countBy = (rows, key) => Object.fromEntries([...groupBy(rows, key)].map(([name, members]) => [name, members.length]));
const summary = {
  version, seed: config.seed, input_identity: inputIdentity, primary_rows: manifest.length, excluded_duplicates: knownDuplicates.map(row => row.sample_id),
  domestic_unique_counts: countBy(domestic, row => row.label), esc50_source_repairs: escRepairs,
  esc50_counts: countBy(esc, row => row.split),
  folds: folds.map(({rows, sampler}, fold) => ({fold, total_counts: countBy(rows, row => row.split),
    domestic_counts: countBy(rows.filter(row => row.target_domain === 'ko_target'), row => `${row.label}/${row.split}`),
    pool_probabilities: Object.fromEntries([...groupBy(sampler, row => row.sampler_pool)].map(([pool, members]) => [pool, members.reduce((sum, row) => sum + row.group_probability, 0)]))})),
  refit_counts: countBy(refit, row => row.split),
  checks: ['audio_paths_exist', 'labels_fixed', 'known_duplicates_verified_and_excluded', 'group_and_original_source_disjoint', 'same_test_across_folds', 'training_sampler_probabilities_normalized', 'test_not_in_sampler'],
  limitations: ['file-class review does not assert event timestamps', 'clip counts do not establish event counts or deployment false alarms per hour', 'source independence is user-confirmed; household/device identities remain unrecorded']
};

fs.mkdirSync(outputDir, {recursive: true});
const backupDir = path.join(root, 'data/raw/private/protocol_backup_20261108_v1');
fs.mkdirSync(backupDir, {recursive: true});
for (const relative of [manifestPath, escPath]) {
  const backup = path.join(backupDir, path.basename(relative));
  if (!fs.existsSync(backup)) fs.copyFileSync(path.join(root, relative), backup);
}
function writeCsv(relative, rows) {
  const headers = [...new Set(rows.flatMap(row => Object.keys(row)))];
  const output = csvText(rows, headers);
  invariant(parseCsv(output).length === rows.length, 'Generated CSV did not round-trip');
  fs.writeFileSync(path.join(root, relative), output, 'utf8');
}
writeCsv(manifestPath, manifest);
writeCsv(escPath, esc);
for (let fold = 0; fold < folds.length; fold++) {
  writeCsv(`data/splits/${version}/fold_${fold}.csv`, folds[fold].rows);
  writeCsv(`data/splits/${version}/sampling_groups_fold_${fold}.csv`, folds[fold].sampler);
}
writeCsv(`data/splits/${version}/refit.csv`, refit);
writeCsv(`data/splits/${version}/sampling_groups_refit.csv`, samplingGroups(refit));
writeCsv(`data/splits/${version}/test.csv`, folds[0].rows.filter(row => row.split === 'test'));
fs.writeFileSync(lockPath, JSON.stringify(lock, null, 2) + '\n');
fs.writeFileSync(path.join(outputDir, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify(summary, null, 2));
