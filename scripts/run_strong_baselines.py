from __future__ import annotations
import sys as _prompt_sys
from pathlib import Path as _PromptPath
for _prompt_parent in _PromptPath(__file__).resolve().parents:
    if (_prompt_parent / 'runtime_prompt_loader.py').is_file():
        if str(_prompt_parent) not in _prompt_sys.path:
            _prompt_sys.path.insert(0, str(_prompt_parent))
        break
from runtime_prompt_loader import get_prompt as _get_prompt
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import pandas as pd
import requests
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'results' / '04_strong_baselines'
RAW_DIR = OUT_DIR / 'raw_outputs'
LOG_DIR = OUT_DIR / 'logs'
API_URL = os.getenv('OPENAI_CHAT_COMPLETIONS_URL', os.getenv('OPENAI_BASE_URL', 'http://localhost:8000/v1').rstrip('/') + '/chat/completions')
DEFAULT_MODEL = os.getenv('OPENAI_MODEL', 'qwen3.5-397b-a17b')
DEFAULT_API_KEY = os.getenv('SEU_API_KEY', os.getenv('OPENAI_API_KEY', ''))
RAG_URL = os.getenv('RAG_RETRIEVAL_URL', 'http://localhost:9380/api/v1/retrieval')
RAG_API_KEY = os.getenv('RAG_API_KEY', '')
RAG_DATASET_ID = os.getenv('RAG_DATASET_ID', '')
BASELINE_CHOICES = ['summarization_only_memory', 'flat_vector_rag', 'time_filtered_flat_rag', 'tree_rag_without_stage_routing', 'full_context_llm', 'no_rag_summary_only', 'flat_rag_fixed_query']
ORIGINAL_RAW_CONTEXT_COLUMNS = ['chartevents_json', 'labevents_json', 'inputevents_json', 'outputevents_json', 'procedureevents_json', 'microbiology_json', 'prescriptions_json', 'pharmacy_json', 'emar_json', 'diagnoses_text_json', 'procedures_text_json']
LEAKAGE_CONTROLLED_RAW_CONTEXT_COLUMNS = ['chartevents_json', 'labevents_json', 'inputevents_json', 'outputevents_json', 'procedureevents_json', 'microbiology_json', 'prescriptions_json', 'pharmacy_json', 'emar_json']
STRICT_NO_SUMMARY_RAW_CONTEXT_COLUMNS = ['chartevents_json', 'labevents_json', 'inputevents_json', 'outputevents_json', 'microbiology_json', 'prescriptions_json', 'pharmacy_json', 'emar_json']
RAW_CONTEXT_COLUMNS = ORIGINAL_RAW_CONTEXT_COLUMNS
CONTEXT_POLICY = 'original'
LEAKAGE_SUMMARY_PATTERNS = ['(:Final|Comprehensive|Overall)(:Judgment|Conclusion|Evaluation)\\s*[::]\\s*(:Existence|Highly suspected|Not supported|Negative|Positive)[^. ;;\\n] {0,80}', '(:AKI|ARDS|Sepsis|Acute Kidney Injury|Acute Respiratory Distress Syndrome|Sepsis)[^. ;;\\n] {0,40} (:Probability|Possibility|Risk)\\s*[::]\\s*[0-9.]+%', '(:Probability|Possibility|Risk)\\s*[::]\\s*[0-9.]+%[^. ;;\\n] {0,40} (:AKI|ARDS|Sepsis|Acute kidney injury|Acute respiratory distress syndrome|Sepsis)']
TASK_SPECS = {'AKI': {'positive': ROOT / 'experiments/public_mimic/vital_tree_rag' / 'aki_info_with_summary_decide_result11.xlsx', 'negative': ROOT / 'experiments/public_mimic/vital_tree_rag' / 'non_aki_info_with_summary_decide_result_judge.xlsx', 'summary_col': 'icu_progress_summary', 'question': 'Does the patient have or have high suspicion for AKI (acute kidney injury)', 'probability_key': 'aki_probability', 'fixed_rag_query': 'Critical Care Medicine Acute Kidney Injury AKI KDIGO Diagnostic Criteria Urine Output Creatinine Risk Assessment Treatment Strategy', 'criteria': 'KDIGO AKI criteria, including 48-hour creatinine rise, 7-day rise from baseline, and oliguria criteria.'}, 'ARDS': {'positive': ROOT / 'experiments/public_mimic/vital_tree_rag' / 'ards_info_with_summary_decide_result11.xlsx', 'negative': ROOT / 'experiments/public_mimic/vital_tree_rag' / 'non_ards_info_with_summary_decide_result_judge.xlsx', 'summary_col': 'icu_progress_summary', 'question': 'Does the patient have or have high suspicion for ARDS (acute respiratory distress syndrome)', 'probability_key': 'ards_probability', 'fixed_rag_query': 'Critical Care Medicine ARDS Acute Respiratory Distress Syndrome Berlin Definition Oxygenation Index PEEP Imaging Exclude Heart Failure', 'criteria': 'Berlin ARDS definition, including acute onset, bilateral pulmonary infiltrates, noncardiogenic respiratory failure, and P/F oxygenation grade.'}, 'Sepsis': {'positive': ROOT / 'experiments/public_mimic/vital_tree_rag' / 'sepsis_info_with_summary_decide_result11.xlsx', 'negative': ROOT / 'experiments/public_mimic/vital_tree_rag' / 'non_sepsis_info_with_summary_decide_result_judge.xlsx', 'summary_col': 'icu_progress_summary', 'question': 'Does the patient have or have high suspicion for sepsis', 'probability_key': 'sepsis_probability', 'fixed_rag_query': 'Critical Care Medicine Sepsis Sepsis-3 Infection SOFA Organ Dysfunction Septic Shock Diagnosis Treatment', 'criteria': 'Sepsis-3 definition includes suspected or confirmed infection and infection-related organ dysfunction, focusing on SOFA/qSOFA.'}}

def ensure_dirs() -> None:
    for path in [OUT_DIR, RAW_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)

def configure_outputs(output_subdir: str, context_policy: str) -> None:
    global OUT_DIR, RAW_DIR, LOG_DIR, CONTEXT_POLICY, RAW_CONTEXT_COLUMNS
    OUT_DIR = ROOT / 'results' / output_subdir
    RAW_DIR = OUT_DIR / 'raw_outputs'
    LOG_DIR = OUT_DIR / 'logs'
    CONTEXT_POLICY = context_policy
    if context_policy == 'strict_no_summary':
        RAW_CONTEXT_COLUMNS = STRICT_NO_SUMMARY_RAW_CONTEXT_COLUMNS
    elif context_policy == 'leakage_controlled':
        RAW_CONTEXT_COLUMNS = LEAKAGE_CONTROLLED_RAW_CONTEXT_COLUMNS
    else:
        RAW_CONTEXT_COLUMNS = ORIGINAL_RAW_CONTEXT_COLUMNS

def normalize_time_column(df: pd.DataFrame) -> pd.Series:
    for col in ['window_end_time', 'chart_day', 'block_end', 'day_end', 'icu_outtime']:
        if col in df.columns:
            return pd.to_datetime(df[col], errors='coerce')
    return pd.Series(range(len(df)), index=df.index)

def compact_value(value: Any, max_chars: int=2200) -> str:
    if pd.isna(value):
        return ''
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return ''
    text = re.sub('\\s+', ' ', text)
    return text[:max_chars]

def scrub_summary_for_leakage(summary: Any, task: str) -> str:
    text = compact_value(summary, 14000)
    if CONTEXT_POLICY == 'strict_no_summary':
        return ''
    if CONTEXT_POLICY != 'leakage_controlled' or not text:
        return text
    cleaned = text
    for pattern in LEAKAGE_SUMMARY_PATTERNS:
        cleaned = re.sub(pattern, '[removed possible target-label leakage]', cleaned, flags=re.IGNORECASE)
    probability_key = TASK_SPECS[task].get('probability_key', '')
    if probability_key:
        cleaned = re.sub(f'{re.escape(probability_key)}\\s*[:=]\\s*[0-9.]+', '[removed possible target-probability leakage]', cleaned, flags=re.IGNORECASE)
    return cleaned[:14000]
STRICT_PATIENT_CONTEXT_PATTERNS = ['(i)\\b(:AKI|ARDS|Sepsis|KDIGO|Berlin|SOFA|qSOFA|ICD)\\b', '(i)\\b(:diagnos(:is|es|ed)|procedure(:s)|operation(:s)|intubation|extubation|ventilation_status)\\b', '(i)\\b(:ards_probability|aki_probability|sepsis_probability)\\s*[:=]\\s*[0-9.]+', '(:acute kidney injury|acute respiratory distress syndrome|sepsis|sepsis|septic shock|diagnosis|diagnosed as|highly suspected|met|satisfied|determined as|considered|possibly|standard|Berlin definition|oliguria criteria|oxygenation index|organ dysfunction|kidney damage)', '(:P/F|PF\\s*ratio|PaO2\\s*/\\s*FiO2|oxygenation index)\\s*[::=]\\s*[0-9.]+']

def scrub_patient_context_for_strict_policy(context: str) -> str:
    if CONTEXT_POLICY != 'strict_no_summary' or not context:
        return context
    cleaned_lines = []
    for line in context.splitlines():
        line = line.strip()
        if not line:
            continue
        if any((re.search(pattern, line) for pattern in STRICT_PATIENT_CONTEXT_PATTERNS)):
            continue
        cleaned_lines.append(line)
    cleaned = '\n'.join(cleaned_lines)
    for pattern in STRICT_PATIENT_CONTEXT_PATTERNS:
        cleaned = re.sub(pattern, '[removed]', cleaned)
    return cleaned[:18000]

def build_raw_context(row: pd.Series, max_chars: int=26000) -> str:
    parts = []
    for col in RAW_CONTEXT_COLUMNS:
        if col not in row.index:
            continue
        text = compact_value(row[col])
        if text:
            parts.append(f'[{col}] {text}')
    context = '\n'.join(parts)
    context = scrub_patient_context_for_strict_policy(context)
    return context[:max_chars]

def load_patient_inputs(task: str) -> pd.DataFrame:
    spec = TASK_SPECS[task]
    frames = []
    for source_name, label in [('positive', 1), ('negative', 0)]:
        df = pd.read_excel(spec[source_name])
        df['_truth_label'] = label
        df['_source_file'] = spec[source_name].name
        df['_row_time'] = normalize_time_column(df)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(['subject_id', 'hadm_id', 'stay_id', '_row_time'], na_position='first')
    patient = data.groupby(['subject_id', 'hadm_id', 'stay_id'], as_index=False).tail(1).copy()
    patient['task'] = task
    patient['case_id'] = task + '_' + patient['subject_id'].astype(str) + '_' + patient['hadm_id'].astype(str) + '_' + patient['stay_id'].astype(str)
    patient['raw_context'] = patient.apply(build_raw_context, axis=1)
    patient['summary_clean'] = patient[spec['summary_col']].map(lambda value: scrub_summary_for_leakage(value, task))
    return patient[['task', 'case_id', 'subject_id', 'hadm_id', 'stay_id', '_truth_label', '_source_file', 'summary_clean', 'raw_context']].rename(columns={'_truth_label': 'truth_label', 'summary_clean': 'summary'})

def retrieve_flat_rag_context(task: str, baseline: str='flat_vector_rag', max_chars: int=5000) -> str:
    spec = TASK_SPECS[task]
    session = requests.Session()
    session.trust_env = False
    question = spec['fixed_rag_query']
    if baseline == 'time_filtered_flat_rag':
        question = f'{question} pre-index early ICU time window diagnosis timing longitudinal change'
    elif baseline == 'tree_rag_without_stage_routing':
        question = f"{question} {spec['criteria']} ICU diagnosis treatment evidence"
    payload = {'question': question, 'dataset_ids': [RAG_DATASET_ID], 'similarity_threshold': 0.3, 'vector_similarity_weight': 0.5, 'page_size': 12}
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {RAG_API_KEY}'}
    response = session.post(RAG_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    chunks = data.get('data', {}).get('chunks', []) if isinstance(data.get('data'), dict) else []
    parts = []
    seen = set()
    for chunk in chunks:
        content = (chunk.get('content') or chunk.get('content_ltks') or '').strip()
        if not content or content in seen:
            continue
        seen.add(content)
        parts.append(content)
    context = '\n\n'.join(parts)
    return context[:max_chars] if context else '(Normal RAG fixed query does not recall available fragments)'

def build_user_prompt(task: str, baseline: str, summary: str, rag_context: str | None, raw_context: str | None=None) -> str:
    return _get_prompt('scripts/run_strong_baselines.py::build_user_prompt', locals())

def build_strong_user_prompt(task: str, baseline: str, summary: str, rag_context: str | None, raw_context: str | None=None) -> str:
    return _get_prompt('scripts/run_strong_baselines.py::build_strong_user_prompt', locals())

def call_local_llm(session: requests.Session, model: str, api_key: str, user_prompt: str, max_retries: int=5) -> tuple[str, dict[str, Any]]:
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {'model': model, 'messages': [{'role': 'system', 'content': _get_prompt('scripts/run_strong_baselines.py::SYSTEM_PROMPT', locals())}, {'role': 'user', 'content': user_prompt}], 'temperature': 0}
    if model.lower().startswith('deepseek'):
        payload['thinking'] = {'type': 'disabled'}
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(API_URL, headers=headers, json=payload, timeout=180)
            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(f'HTTP {response.status_code}: {response.text[:500]}')
            data = response.json()
            content = data['choices'][0]['message']['content'].strip()
            usage = data.get('usage') or {}
            return (content, usage)
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(min(45, 2 ** attempt))
    return (json.dumps({'probability': None, 'label': None, 'rationale': f'LLM_ERROR {last_error}'}, ensure_ascii=False), {})

def coerce_probability(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        prob = float(value)
        return max(0.0, min(1.0, prob))
    text = '' if value is None else str(value)
    text = text.replace('-', '0').replace('ﾂ', '')
    match = re.search('(<!\\d)(0\\.\\d+|[01](:\\.\\d+))(!\\d)', text)
    if not match:
        return None
    token = match.group(1)
    prob = float('0' + token if token.startswith('.') else token)
    return max(0.0, min(1.0, prob))

def parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub('^```(:json)', '', text).strip()
        text = re.sub('```$', '', text).strip()
    match = re.search('\\{.*\\}', text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except Exception:
        normalized_raw = raw.replace('-', '0').replace('ﾂ', '')
        prob_match = re.search('"probability"\\s*:\\s*[^0-9.]{0,80}(0\\.\\d+|[01](:\\.\\d+))', normalized_raw, flags=re.IGNORECASE)
        label_match = re.search('"label"\\s*:\\s*([01])', raw, flags=re.IGNORECASE)
        if prob_match:
            token = prob_match.group(1)
            prob = float('0' + token if token.startswith('.') else token)
        else:
            prob = None
        label = int(label_match.group(1)) if label_match else int(prob >= 0.5) if prob is not None else None
        return {'probability': prob, 'label': label, 'rationale': raw[:500]}
    prob = coerce_probability(data.get('probability'))
    label = data.get('label')
    try:
        label = int(label)
    except Exception:
        label = int(prob >= 0.5) if prob is not None else None
    if label not in (0, 1):
        label = int(prob >= 0.5) if prob is not None else None
    return {'probability': prob, 'label': label, 'rationale': str(data.get('rationale', ''))[:500]}

def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get('case_id') and coerce_probability(row.get('probability')) is not None and (row.get('label') is not None):
                done.add(row['case_id'])
    return done

def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def run_baseline(task: str, baseline: str, model: str, api_key: str, workers: int, limit: int | None) -> Path:
    ensure_dirs()
    output_path = RAW_DIR / f"{baseline}_{task.lower()}_{model.replace('/', '_')}.jsonl"
    log_path = LOG_DIR / f"{baseline}_{task.lower()}_{model.replace('/', '_')}.log"
    data = load_patient_inputs(task)
    if limit is not None:
        data = data.head(limit)
    done = load_done(output_path)
    pending = data[~data['case_id'].isin(done)].copy()
    rag_context = None
    if baseline in {'flat_rag_fixed_query', 'flat_vector_rag', 'time_filtered_flat_rag', 'tree_rag_without_stage_routing'}:
        rag_context = retrieve_flat_rag_context(task, baseline)
        (OUT_DIR / f'{baseline}_context_{task.lower()}.txt').write_text(rag_context, encoding='utf-8')
    start_line = f'{pd.Timestamp.now().isoformat()} task={task} baseline={baseline} model={model} total={len(data)} done={len(done)} pending={len(pending)}\n'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write(start_line)
    lock = threading.Lock()
    progress = {'n': 0}

    def worker(row: pd.Series) -> None:
        session = requests.Session()
        session.trust_env = False
        prompt = build_strong_user_prompt(task, baseline, row['summary'], rag_context, row.get('raw_context'))
        raw, usage = call_local_llm(session, model, api_key, prompt)
        parsed = parse_json_response(raw)
        out = {'baseline': baseline, 'task': task, 'context_policy': CONTEXT_POLICY, 'case_id': row['case_id'], 'subject_id': int(row['subject_id']), 'hadm_id': int(row['hadm_id']), 'stay_id': int(row['stay_id']), 'truth_label': int(row['truth_label']), 'probability': parsed['probability'], 'label': parsed['label'], 'rationale': parsed['rationale'], 'model': model, 'usage': usage, 'raw_response': raw}
        append_jsonl(output_path, out, lock)
        with lock:
            progress['n'] += 1
            if progress['n'] % 10 == 0 or progress['n'] == len(pending):
                msg = f"{pd.Timestamp.now().isoformat()} completed {progress['n']}/{len(pending)}\n"
                with log_path.open('a', encoding='utf-8') as f:
                    f.write(msg)
                print(msg.strip(), flush=True)
    if len(pending) == 0:
        print(f'Already complete: {output_path}')
        return output_path
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, row) for _, row in pending.iterrows()]
        for future in as_completed(futures):
            future.result()
    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(description='Run local-LLM strong baselines on MIMIC patient summaries.')
    parser.add_argument('--task', choices=sorted(TASK_SPECS), required=True)
    parser.add_argument('--baseline', choices=BASELINE_CHOICES, required=True)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--api-key', default=DEFAULT_API_KEY)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--context-policy', choices=['original', 'leakage_controlled', 'strict_no_summary'], default='original', help='original keeps previous inputs; leakage_controlled removes obvious target-label statements; strict_no_summary removes generated summaries and stricter target-diagnosis cues from patient context.')
    parser.add_argument('--output-subdir', default=None, help='Subdirectory under results/. Defaults to 04_strong_baselines for original and 04b_leakage_controlled_strong_baselines for leakage_controlled.')
    args = parser.parse_args()
    output_subdir = args.output_subdir or ('04_strong_baselines_strict' if args.context_policy == 'strict_no_summary' else '04b_leakage_controlled_strong_baselines' if args.context_policy == 'leakage_controlled' else '04_strong_baselines')
    configure_outputs(output_subdir, args.context_policy)
    path = run_baseline(args.task, args.baseline, args.model, args.api_key, args.workers, args.limit)
    print(f'Wrote {path}')
if __name__ == '__main__':
    main()
