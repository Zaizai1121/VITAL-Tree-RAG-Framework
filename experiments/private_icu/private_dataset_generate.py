import sys as _prompt_sys
from pathlib import Path as _PromptPath
for _prompt_parent in _PromptPath(__file__).resolve().parents:
    if (_prompt_parent / 'runtime_prompt_loader.py').is_file():
        if str(_prompt_parent) not in _prompt_sys.path:
            _prompt_sys.path.insert(0, str(_prompt_parent))
        break
from runtime_prompt_loader import get_prompt as _get_prompt
import sys
from pathlib import Path
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, local
from typing import Any, Dict, List, Optional
import requests
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None
try:
    import pymysql
except ModuleNotFoundError:
    pymysql = None
SCRIPT_DIR = Path(__file__).resolve().parent
DB_CONFIG = {'host': os.getenv('PRIVATE_TEST_DB_HOST', 'localhost'), 'port': int(os.getenv('PRIVATE_TEST_DB_PORT', '3306')), 'user': os.getenv('PRIVATE_TEST_DB_USER', 'root'), 'password': os.getenv('PRIVATE_TEST_DB_PASSWORD', ''), 'database': os.getenv('PRIVATE_TEST_DB_NAME', 'chat_history02')}
if pymysql is not None:
    DB_CONFIG['cursorclass'] = pymysql.cursors.DictCursor
BASE_URL = os.getenv('OPENAI_BASE_URL', 'http://localhost:8000/v1')
BASE_MODEL_CONFIGS = [{'name': 'qwen3.5-397b-a17b', 'api_key': os.getenv('PRIVATE_TEST_QWEN35_API_KEY', ''), 'slug': 'qwen3_5_397b_a17b'}, {'name': 'DeepSeek-V4-Pro', 'api_key': os.getenv('PRIVATE_TEST_DEEPSEEK_V4_API_KEY', ''), 'slug': 'deepseek_v4_pro'}]
DEFAULT_MODEL_CONFIG = BASE_MODEL_CONFIGS[0]
SMART_MODEL_NAME = os.getenv('PRIVATE_TEST_SMART_MODEL_NAME', DEFAULT_MODEL_CONFIG['name'])
SMART_API_KEY = os.getenv('PRIVATE_TEST_SMART_API_KEY', DEFAULT_MODEL_CONFIG['api_key'])
RAG_API_BASE = os.getenv('PRIVATE_TEST_RAG_API_BASE', 'http://localhost:9380')
RAG_API_KEY = os.getenv('PRIVATE_TEST_RAG_API_KEY', '')
RAG_DATASET_ID = os.getenv('PRIVATE_TEST_RAG_DATASET_ID', '')
RAG_SIMILARITY_THRESHOLD = float(os.getenv('PRIVATE_TEST_RAG_SIMILARITY_THRESHOLD', '0.5'))
RAG_VECTOR_SIM_WEIGHT = float(os.getenv('PRIVATE_TEST_RAG_VECTOR_SIM_WEIGHT', '0.5'))
RAG_TOP_K = int(os.getenv('PRIVATE_TEST_RAG_TOP_K', '10'))
DEFAULT_NUM_PATIENTS = int(os.getenv('PRIVATE_TEST_NUM_PATIENTS', '0'))
DEFAULT_MAX_WORKERS = int(os.getenv('PRIVATE_TEST_MAX_WORKERS', '5'))
DEFAULT_MODEL_WORKERS = int(os.getenv('PRIVATE_TEST_MODEL_WORKERS', '3'))
DEFAULT_MAX_RAW_CHARS = int(os.getenv('PRIVATE_TEST_MAX_RAW_CHARS', '30000'))
DEFAULT_STAGE = os.getenv('PRIVATE_TEST_INDEX_STAGE', 'admission')
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / 'private_dataset_model_outputs.xlsx'
HTTP_STATE = local()

def _get_thread_session(name: str) -> requests.Session:
    session = getattr(HTTP_STATE, name, None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        setattr(HTTP_STATE, name, session)
    return session

class ProgressTracker:

    def __init__(self, total_patients: int, total_responses: int):
        self.total_patients = total_patients
        self.total_responses = total_responses
        self.completed_patients = 0
        self.completed_responses = 0
        self.started_at = time.time()
        self.lock = Lock()

    def _elapsed(self) -> str:
        seconds = int(time.time() - self.started_at)
        return f'{seconds // 60:02d}:{seconds % 60:02d}'

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        return f'{minutes:02d}:{seconds:02d}'

    def _eta(self, completed: int, total: int) -> str:
        if completed <= 0 or total <= 0:
            return '--:--'
        elapsed = time.time() - self.started_at
        remaining = elapsed / completed * (total - completed)
        return self._format_duration(remaining)

    @staticmethod
    def _percent(completed: int, total: int) -> float:
        if total <= 0:
            return 100.0
        return completed / total * 100

    def stage(self, message: str) -> None:
        with self.lock:
            print(f'[{self._elapsed()}] {message}', flush=True)

    def response_done(self, patient_id, model_name: str, mode_key: str) -> None:
        with self.lock:
            self.completed_responses += 1
            print(f'[{self._elapsed()}] Answer progress{self.completed_responses}/{self.total_responses} ({self._percent(self.completed_responses, self.total_responses):.1f}%, ETA {self._eta(self.completed_responses, self.total_responses)}) | patient={patient_id} | model={model_name} | mode={mode_key}', flush=True)

    def patient_done(self, patient_id) -> None:
        with self.lock:
            self.completed_patients += 1
            print(f'[{self._elapsed()}] Patient Progress{self.completed_patients}/{self.total_patients} ({self._percent(self.completed_patients, self.total_patients):.1f}%, ETA {self._eta(self.completed_patients, self.total_patients)}) | patient={patient_id}Finish', flush=True)

def _connect_db():
    if pymysql is None:
        raise RuntimeError('The dependency pymysql is missing, please install it first or switch to a Python environment that already has dependencies.')
    return pymysql.connect(**DB_CONFIG)

def _safe_json_loads(text: str, default=None):
    try:
        return json.loads(text)
    except Exception:
        return default

def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    direct = _safe_json_loads(text)
    if isinstance(direct, dict):
        return direct
    match = re.search('\\{.*\\}', text, flags=re.S)
    if match:
        parsed = _safe_json_loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    return None

def call_llm(system_prompt: str, user_prompt: str, temperature: float=0.0, model: Optional[str]=None, api_key: Optional[str]=None, retries: int=3) -> str:
    use_model = model or DEFAULT_MODEL_CONFIG['name']
    use_key = api_key if api_key is not None else DEFAULT_MODEL_CONFIG['api_key']
    if not use_key:
        return f'<<LLM_ERROR: missing api key for {use_model}>>'
    url = f'{BASE_URL}/chat/completions'
    headers = {'Authorization': f'Bearer {use_key}', 'Content-Type': 'application/json'}
    payload = {'model': use_model, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}], 'temperature': temperature}
    last_error = ''
    for attempt in range(retries):
        try:
            resp = _get_thread_session('llm_session').post(url, headers=headers, json=payload, timeout=360)
            if resp.status_code == 200:
                data = resp.json()
                return data['choices'][0]['message']['content'].strip()
            last_error = f'status={resp.status_code}, body={resp.text[:300]}'
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(2 * (attempt + 1))
    return f'<<LLM_ERROR: {last_error}>>'

def call_text_llm(system_prompt: str, user_prompt: str, model: str, api_key: str) -> str:
    return call_llm(system_prompt, user_prompt, temperature=0.0, model=model, api_key=api_key)

def ragflow_retrieval(question: str) -> Dict[str, Any]:
    if not RAG_API_KEY:
        return {'data': []}
    url = f'{RAG_API_BASE}/api/v1/retrieval'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {RAG_API_KEY}'}
    payload = {'question': question, 'dataset_ids': [RAG_DATASET_ID], 'similarity_threshold': RAG_SIMILARITY_THRESHOLD, 'vector_similarity_weight': RAG_VECTOR_SIM_WEIGHT, 'top_k': RAG_TOP_K}
    try:
        resp = _get_thread_session('rag_session').post(url, headers=headers, json=payload, timeout=30)
        return resp.json()
    except Exception:
        return {'data': []}

def format_rag_context(rag_json: Dict[str, Any]) -> str:
    if not isinstance(rag_json, dict):
        return ''
    data = rag_json.get('data')
    target_list: List[Any] = []
    if isinstance(data, list):
        target_list = data[0] if data and isinstance(data[0], list) else data
    elif isinstance(data, dict):
        if isinstance(data.get('chunks'), list):
            target_list = data['chunks']
        elif isinstance(data.get('results'), list):
            target_list = data['results']
        else:
            for value in data.values():
                if isinstance(value, list):
                    target_list = value
                    break
    parts = []
    for idx, item in enumerate(target_list[:RAG_TOP_K]):
        if not isinstance(item, dict):
            continue
        text = item.get('content_with_weight') or item.get('content') or item.get('text') or ''
        if text:
            parts.append(f'[{idx + 1}] {str(text)[:700]}')
    return '\n\n'.join(parts) if parts else '(No valid search evidence was recalled)'

def summarize_rag_context(raw_context: str) -> str:
    if not raw_context.strip() or 'No valid search evidence was recalled' in raw_context:
        return '(The research system did not provide valid search evidence)'
    summary = call_llm(_get_prompt('EVIDENCE_SYNTHESIS_SYSTEM_PROMPT', locals()), _get_prompt('experiments/private_icu/private_dataset_generate.py::inline_L288', locals()), temperature=0.1, model=SMART_MODEL_NAME, api_key=SMART_API_KEY, retries=2)
    if summary.startswith('<<LLM_ERROR'):
        return raw_context
    return summary

def extract_keywords(text: str) -> Dict[str, List[str]]:
    raw = call_llm(_get_prompt('RETRIEVAL_TERM_SYSTEM_PROMPT', locals()), f'Case information: \n{text[:3000]}', temperature=0.0, model=SMART_MODEL_NAME, api_key=SMART_API_KEY, retries=2)
    parsed = _parse_json_from_text(raw)
    if not isinstance(parsed, dict):
        return {'level1': [], 'level2': [], 'level3': []}
    return {'level1': parsed.get('level1', []) if isinstance(parsed.get('level1', []), list) else [], 'level2': parsed.get('level2', []) if isinstance(parsed.get('level2', []), list) else [], 'level3': parsed.get('level3', []) if isinstance(parsed.get('level3', []), list) else []}

def execute_advanced_rag(case_text: str) -> str:
    if not RAG_API_KEY:
        return '(RAG is enabled in this mode, but PRIVATE_TEST_RAG_API_KEY is not configured and no retrieval is performed)'
    keywords = extract_keywords(case_text)
    query_parts = keywords['level1'] + keywords['level2'] + keywords['level3']
    query = ' '.join((str(part) for part in query_parts if str(part).strip()))
    if not query:
        query = 'ICU auxiliary diagnosis treatment recommendations monitoring'
    raw_context = format_rag_context(ragflow_retrieval(query))
    return summarize_rag_context(raw_context)

def execute_a_optimized_rag(case_text: str) -> str:
    if not RAG_API_KEY:
        return '(RAG is enabled in this mode, but PRIVATE_TEST_RAG_API_KEY is not configured and no retrieval is performed)'
    keywords = extract_keywords(case_text)
    base_terms = keywords['level1'] + keywords['level2'] + keywords['level3']
    base_query = ' '.join((str(part) for part in base_terms if str(part).strip()))
    if not base_query:
        base_query = 'ICU Auxiliary Diagnosis Treatment Recommendations Monitoring Risk Review'
    focused_queries = [f'{base_query}ICU Diagnostic Criteria Differential Diagnosis Severity Stratification', f'{base_query}ICU treatment strategies, contraindications, organ function adjustments', f'{base_query}Disease change trend identification early warning indicators monitoring', f'{base_query}Critical Care Medicine Guidelines Consensus Safety Handover']
    contexts = []
    for query in focused_queries:
        raw_context = format_rag_context(ragflow_retrieval(query))
        if raw_context and 'No valid search evidence was recalled' not in raw_context:
            contexts.append(f'[Search question]{query}\n{raw_context}')
    if not contexts:
        return summarize_rag_context(format_rag_context(ragflow_retrieval(base_query)))
    combined = '\n\n'.join(contexts)[:12000]
    return call_llm(_get_prompt('EVIDENCE_SYNTHESIS_SYSTEM_PROMPT', locals()), _get_prompt('experiments/private_icu/private_dataset_generate.py::inline_L359', locals()), temperature=0.0, model=SMART_MODEL_NAME, api_key=SMART_API_KEY, retries=2)

def get_test_patients(limit: int) -> List[Any]:
    patients_file = SCRIPT_DIR / 'testid.json'
    if not patients_file.exists():
        raise FileNotFoundError(f'not found{patients_file}')
    text = patients_file.read_text(encoding='utf-8').strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            patient_ids = loaded
        elif isinstance(loaded, dict):
            values = loaded.get('patient_id') or loaded.get('patient_ids') or loaded.get('patients')
            if isinstance(values, list):
                patient_ids = values
            else:
                raise ValueError('JSON object must contain patient_id/patient_ids/patients list')
        else:
            raise ValueError('Unsupported JSON format')
    except Exception:
        patient_ids = [line.strip() for line in text.splitlines() if line.strip()]
    return patient_ids[:limit] if limit and limit > 0 else patient_ids

def row_to_text(row: Optional[Dict[str, Any]], excluded: Optional[List[str]]=None) -> str:
    if not row:
        return ''
    excluded_set = set(excluded or [])
    parts = []
    for key, value in row.items():
        if key in excluded_set or value is None or value == '':
            continue
        parts.append(f'{key}: {value}')
    return '\n'.join(parts)

def get_pre_summary(cursor, patient_id) -> str:
    cursor.execute('SELECT pre_summary FROM pre_patient_info WHERE patient_id = %s LIMIT 1', (patient_id,))
    row = cursor.fetchone()
    return row['pre_summary'] if row and row.get('pre_summary') else ''

def get_pre_raw(cursor, patient_id) -> str:
    cursor.execute('SELECT * FROM pre_patient_info WHERE patient_id = %s LIMIT 1', (patient_id,))
    row = cursor.fetchone()
    if not row:
        return ''
    fields = ['pre_exams', 'pre_check', 'pre_surgery_record', 'pre_pathological_record']
    parts = ['【Before admission】']
    for field in fields:
        value = row.get(field)
        if value:
            parts.append(f'{field}: {value}')
    return '\n'.join(parts)

def get_raw_history_until_date(cursor, patient_id, end_date: str, max_chars: int) -> str:
    parts = []
    pre_raw = get_pre_raw(cursor, patient_id)
    if pre_raw:
        parts.append(pre_raw)
    current_chars = 0
    cursor.execute('SELECT * FROM patient_info WHERE patient_id = %s AND date <= %s ORDER BY date ASC', (patient_id, end_date))
    rows = cursor.fetchall()
    cols = ['patient_info', 'examine', 'doctor_advice', 'inspection_visit', 'surgery_record', 'monitor']
    for row in rows:
        if current_chars >= max_chars:
            break
        parts.append(f"\n [Date:{row.get('date')}】")
        for col in cols:
            value = str(row.get(col) or '').strip()
            if not value:
                continue
            entry = f'[{col}]: {value}'
            parts.append(entry)
            current_chars += len(entry)
    return '\n'.join((part for part in parts if part)).strip()[:max_chars]

def get_admission_raw_data(cursor, patient_id, max_chars: int) -> str:
    text_buffer = []
    current_chars = 0
    cursor.execute('SELECT * FROM pre_patient_info WHERE patient_id = %s LIMIT 1', (patient_id,))
    pre = cursor.fetchone()
    if pre:
        text_buffer.append('[Original record before admission]')
        for key in ['pre_exams', 'pre_check', 'pre_surgery_record', 'pre_pathological_record']:
            if pre.get(key):
                text_buffer.append(f'{key}: {pre[key]}')
        text_buffer.append('-' * 20)
    cursor.execute('SELECT * FROM patient_info WHERE patient_id = %s ORDER BY date ASC LIMIT 1', (patient_id,))
    row = cursor.fetchone()
    if row:
        text_buffer.append(f"[First day of admission:{row['date']}】")
        for key in ['patient_info', 'examine', 'doctor_advice', 'inspection_visit', 'surgery_record', 'monitor']:
            if row.get(key):
                value = str(row[key])
                text_buffer.append(f'[{key}]: {value}')
                current_chars += len(value)
                if current_chars >= max_chars:
                    break
    return '\n'.join(text_buffer)[:max_chars]

def get_admission_case(cursor, patient_id, max_raw_chars: int) -> Optional[Dict[str, str]]:
    pre_summary = get_pre_summary(cursor, patient_id)
    cursor.execute('SELECT * FROM patient_info WHERE patient_id = %s ORDER BY date ASC LIMIT 1', (patient_id,))
    first_day = cursor.fetchone()
    if not first_day:
        return None
    index_time = str(first_day['date'])
    current_status = row_to_text(first_day, excluded=['id', 'patient_id'])
    raw_data = get_admission_raw_data(cursor, patient_id, max_raw_chars)
    return {'index_time': index_time, 'cur_date': index_time, 'pre_summary': pre_summary, 'cur_status': current_status, 'case_context': current_status, 'raw_data': raw_data}

def get_middle_case(cursor, patient_id, max_raw_chars: int) -> Optional[Dict[str, str]]:
    pre_summary = get_pre_summary(cursor, patient_id)
    cursor.execute('SELECT * FROM patient_summary WHERE patient_id = %s ORDER BY update_time ASC', (patient_id,))
    summaries = cursor.fetchall()
    if not summaries:
        return None
    target = summaries[len(summaries) // 2]
    index_time = str(target['update_time'])
    summary = target.get('summary', '')
    raw_data = get_raw_history_until_date(cursor, patient_id, index_time, max_raw_chars)
    return {'index_time': index_time, 'cur_date': index_time, 'pre_summary': pre_summary, 'cur_status': summary, 'case_context': summary, 'raw_data': raw_data}

def get_last_case(cursor, patient_id, max_raw_chars: int) -> Optional[Dict[str, str]]:
    pre_summary = get_pre_summary(cursor, patient_id)
    cursor.execute('SELECT * FROM patient_info WHERE patient_id = %s ORDER BY date DESC LIMIT 1', (patient_id,))
    last_row = cursor.fetchone()
    if not last_row:
        return None
    index_time = str(last_row['date'])
    current_status = row_to_text(last_row, excluded=['id', 'patient_id'])
    cursor.execute('SELECT summary FROM patient_summary WHERE patient_id = %s AND update_time <= %s ORDER BY update_time DESC LIMIT 1', (patient_id, index_time))
    summary_row = cursor.fetchone()
    summary = summary_row['summary'] if summary_row and summary_row.get('summary') else 'No summary of disease history before index time'
    raw_data = get_raw_history_until_date(cursor, patient_id, index_time, max_raw_chars)
    return {'index_time': index_time, 'pre_summary': pre_summary, 'case_context': f'【Summary of disease course before index time】 \n{summary}\n \n [index time current information] \n{current_status}', 'raw_data': raw_data}

def get_case_data(patient_id, stage: str, max_raw_chars: int) -> Optional[Dict[str, str]]:
    conn = _connect_db()
    try:
        with conn.cursor() as cursor:
            if stage == 'admission':
                return get_admission_case(cursor, patient_id, max_raw_chars)
            if stage == 'middle':
                return get_middle_case(cursor, patient_id, max_raw_chars)
            if stage == 'last':
                return get_last_case(cursor, patient_id, max_raw_chars)
            raise ValueError(f'Unknown stage:{stage}')
    finally:
        conn.close()

def build_user_prompt(mode_label: str, index_time: str, pre_summary: str, case_context: str, retrieved_contents: str, raw_data: str) -> str:
    return _get_prompt('experiments/private_icu/private_dataset_generate.py::build_user_prompt', locals())

def process_patient(patient_id, stage: str, max_raw_chars: int, model_workers: int, progress: Optional[ProgressTracker]=None) -> Dict[str, Any]:
    if progress:
        progress.stage(f'patient={patient_id} start: load case data')
    case_data = get_case_data(patient_id, stage, max_raw_chars)
    if not case_data:
        return {'patient_id': patient_id, 'index_stage': stage, 'error': 'No data'}
    index_time = case_data['index_time']
    cur_date = case_data.get('cur_date', index_time)
    pre_summary = case_data['pre_summary']
    case_context = case_data['case_context']
    raw_data = case_data['raw_data']
    result: Dict[str, Any] = {'patient_id': patient_id, 'index_stage': stage, 'index_time': index_time, 'input_date': cur_date}
    rag_input = f'{pre_summary}\n{case_context}'
    if progress:
        progress.stage(f'patient={patient_id} start: RAG retrieval/summary')
    rag_content = execute_a_optimized_rag(rag_input)
    if progress:
        progress.stage(f'patient={patient_id} done: RAG, start model calls')
    mode_prompts = {'A_standard': build_user_prompt('A_standard_summary_memory_with_rag', index_time, pre_summary=pre_summary, case_context=case_context, retrieved_contents=rag_content, raw_data=_get_prompt('experiments/private_icu/private_dataset_generate.py::inline_L624', locals())), 'B_no_rag': build_user_prompt('B_summary_memory_without_rag', index_time, pre_summary=pre_summary, case_context=case_context, retrieved_contents=_get_prompt('experiments/private_icu/private_dataset_generate.py::inline_L631', locals()), raw_data=_get_prompt('experiments/private_icu/private_dataset_generate.py::inline_L632', locals())), 'C_raw': build_user_prompt('C_raw_records_only', index_time, pre_summary=_get_prompt('experiments/private_icu/private_dataset_generate.py::inline_L637', locals()), case_context=_get_prompt('experiments/private_icu/private_dataset_generate.py::inline_L638', locals()), retrieved_contents=_get_prompt('experiments/private_icu/private_dataset_generate.py::inline_L639', locals()), raw_data=raw_data)}
    call_jobs = []
    for model_config in BASE_MODEL_CONFIGS:
        for mode_key, prompt in mode_prompts.items():
            call_jobs.append((model_config, mode_key, prompt))
    with ThreadPoolExecutor(max_workers=max(1, model_workers)) as executor:
        futures = {}
        for model_config, mode_key, prompt in call_jobs:
            column = f"response_{model_config['slug']}_{mode_key}"
            future = executor.submit(call_text_llm, _get_prompt('PRIVATE_CLINICAL_RESPONSE_SYSTEM_PROMPT', locals()), prompt, model_config['name'], model_config['api_key'])
            futures[future] = (column, model_config['name'], mode_key)
        for future in as_completed(futures):
            column, model_name, mode_key = futures[future]
            try:
                result[column] = future.result()
            except Exception as exc:
                result[column] = f'<<LLM_ERROR: {repr(exc)}>>'
            if progress:
                progress.response_done(patient_id, model_name, mode_key)
    return result

def save_results(results: List[Dict[str, Any]], output_file: Path) -> None:
    if pd is None:
        raise RuntimeError('The dependency pandas/openpyxl is missing and cannot be written to Excel. Please install it first or switch to a Python environment that already has dependencies.')
    df = pd.DataFrame(results)
    desired_order = ['patient_id', 'index_stage', 'index_time', 'input_date']
    for model_config in BASE_MODEL_CONFIGS:
        desired_order.extend([f"response_{model_config['slug']}_A_standard", f"response_{model_config['slug']}_B_no_rag", f"response_{model_config['slug']}_C_raw"])
    desired_order.append('error')
    final_cols = [col for col in desired_order if col in df.columns]
    final_cols += [col for col in df.columns if col not in final_cols]
    df[final_cols].to_excel(output_file, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate private ICU dataset model outputs.')
    parser.add_argument('--stage', choices=['admission', 'middle', 'last'], default=DEFAULT_STAGE)
    parser.add_argument('--num-patients', type=int, default=DEFAULT_NUM_PATIENTS)
    parser.add_argument('--workers', type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument('--model-workers', type=int, default=DEFAULT_MODEL_WORKERS)
    parser.add_argument('--max-raw-chars', type=int, default=DEFAULT_MAX_RAW_CHARS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    output_file = args.output if args.output.is_absolute() else SCRIPT_DIR / args.output
    patient_ids = get_test_patients(args.num_patients)
    total_responses = len(patient_ids) * len(BASE_MODEL_CONFIGS) * 3
    progress = ProgressTracker(len(patient_ids), total_responses)
    print(f'Start generating private dataset model output: stage={args.stage}, patients={len(patient_ids)}, response_tasks={total_responses}, models={len(BASE_MODEL_CONFIGS)}, modes=3, workers={args.workers}, model_workers={args.model_workers}')
    print(f'Concurrency settings:Maximum simultaneous processing{max(1, args.workers)}patients;Each patient can initiate at most{max(1, args.model_workers)}A model answers the request.')
    print(f'Output file:{output_file}')
    results_by_index: Dict[int, Dict[str, Any]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(process_patient, patient_id, args.stage, args.max_raw_chars, args.model_workers, progress): (idx, patient_id) for idx, patient_id in enumerate(patient_ids)}
        for future in as_completed(futures):
            idx, patient_id = futures[future]
            try:
                results_by_index[idx] = future.result()
            except Exception as exc:
                results_by_index[idx] = {'patient_id': patient_id, 'index_stage': args.stage, 'error': repr(exc)}
            completed += 1
            progress.patient_done(patient_id)
            if completed % args.checkpoint_every == 0 or completed == len(patient_ids):
                ordered_results = [results_by_index[i] for i in range(len(patient_ids)) if i in results_by_index]
                save_results(ordered_results, output_file)
                progress.stage(f'checkpoint saved: {completed}/{len(patient_ids)} -> {output_file}')
    ordered_results = [results_by_index[i] for i in range(len(patient_ids)) if i in results_by_index]
    save_results(ordered_results, output_file)
    print(f'Finish:{output_file}')
if __name__ == '__main__':
    main()
