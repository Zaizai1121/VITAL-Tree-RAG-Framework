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
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
import os
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
INPUT_FILE = 'sepsis_info_with_summary.xlsx'
OUTPUT_FILE = 'sepsis_info_with_summary_decide11.xlsx'
BASE_URL = os.getenv('OPENAI_BASE_URL', 'http://localhost:8000/v1')
MODEL_NAME = 'qwen3-235b-thinking-2507-bigdata'
SEU_API_KEY = os.getenv('SEU_API_KEY', os.getenv('OPENAI_API_KEY', ''))
SMART_MODEL_NAME = 'DeepSeek-V3.1'
SMART_API_KEY = os.getenv('SMART_API_KEY', os.getenv('SEU_API_KEY', os.getenv('OPENAI_API_KEY', '')))
if not SEU_API_KEY:
    raise RuntimeError('SEU API Key not found, please set SEU_API_KEY or fill it in the script.')
RAG_API_BASE = os.getenv('RAG_API_BASE', 'http://localhost:9380')
RAG_API_KEY = os.getenv('RAG_API_KEY', '')
RAG_DATASET_ID = os.getenv('RAG_DATASET_ID', '')
RAG_SIMILARITY_THRESHOLD = 0.45
RAG_VECTOR_SIM_WEIGHT = 0.5
RAG_TOP_K = 10
MAX_RETRIES = 6
RETRY_BASE_SLEEP = 2.0
RETRY_MAX_SLEEP = 60.0
CHECKPOINT_EVERY_N_GROUPS = 5
CHECKPOINT_FILE = OUTPUT_FILE.replace('.xlsx', '.checkpoint.xlsx')
SEU_SESSION = requests.Session()
SEU_SESSION.trust_env = False
RAG_SESSION = requests.Session()
RAG_SESSION.trust_env = False

def _safe_json_loads(s: str, default=None):
    try:
        return json.loads(s)
    except Exception:
        return default

def _safe_parse_json_response(resp: requests.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
        if data is None:
            return {'_raw_text': None, '_status_code': resp.status_code}
        return data
    except Exception:
        try:
            txt = resp.text
        except Exception:
            txt = None
        return {'_raw_text': txt, '_status_code': getattr(resp, 'status_code', None)}

def call_seu_llm(system_prompt: str, user_prompt: str, temperature: float=0.0, model: str=None, api_key: str=None) -> str:
    """Call the SEU large model interface (with retry) and return text results."""
    use_model = model if model else MODEL_NAME
    use_key = api_key if api_key else SEU_API_KEY
    url = f'{BASE_URL}/chat/completions'
    headers = {'Authorization': f'Bearer {use_key}', 'Content-Type': 'application/json'}
    payload = {'model': use_model, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}], 'temperature': temperature}
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SEU_SESSION.post(url, headers=headers, data=json.dumps(payload), timeout=500)
            if not 200 <= resp.status_code < 300:
                raise RuntimeError(f'HTTP {resp.status_code}, body={_safe_parse_json_response(resp)}')
            data = _safe_parse_json_response(resp)
            if isinstance(data, dict) and data.get('choices'):
                c0 = data['choices'][0]
                msg = c0.get('message') if isinstance(c0, dict) else None
                if isinstance(msg, dict) and msg.get('content') is not None:
                    return str(msg['content']).strip()
                if isinstance(c0, dict) and c0.get('text') is not None:
                    return str(c0['text']).strip()
            raise RuntimeError(f'The return structure is not as expected:{data}')
        except Exception as e:
            last_err = e
            sleep_s = min(RETRY_MAX_SLEEP, RETRY_BASE_SLEEP * 2 ** (attempt - 1))
            print(f'[SEU LLM retry ({use_model})] attempt={attempt}/{MAX_RETRIES}, error={e}, sleep={sleep_s:.1f}s')
            time.sleep(sleep_s)
    return f'<<LLM_ERROR>> {str(last_err)}'

def ragflow_retrieval(question: str) -> Dict[str, Any]:
    """RAGFlow retrieval (with retry), returns json."""
    url = f'{RAG_API_BASE}/api/v1/retrieval'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {RAG_API_KEY}'}
    payload = {'question': question, 'dataset_ids': [RAG_DATASET_ID], 'similarity_threshold': RAG_SIMILARITY_THRESHOLD, 'vector_similarity_weight': RAG_VECTOR_SIM_WEIGHT, 'top_k': RAG_TOP_K}
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = RAG_SESSION.post(url, headers=headers, json=payload, timeout=60)
            if not 200 <= r.status_code < 300:
                raise RuntimeError(f'HTTP {r.status_code}, body={_safe_parse_json_response(r)}')
            return _safe_parse_json_response(r)
        except Exception as e:
            last_err = e
            sleep_s = min(RETRY_MAX_SLEEP, RETRY_BASE_SLEEP * 2 ** (attempt - 1))
            print(f'[RAGFlow retry] attempt={attempt}/{MAX_RETRIES}, error={e}, sleep={sleep_s:.1f}s')
            time.sleep(sleep_s)
    return {'_error': str(last_err), 'data': []}

def extract_3level_keywords(summary_text: str) -> Dict[str, List[str]]:
    user_prompt = _get_prompt('experiments/public_mimic/vital_tree_rag/decide_sepsis_p.py::inline_L164', locals())
    print('\n' + '=' * 30 + ' [KEYWORD EXTRACTION PROMPT] ' + '=' * 30)
    print(f'Model: {SMART_MODEL_NAME}')
    print(f"--- SYSTEM PROMPT ---\n{_get_prompt('RETRIEVAL_TERM_SYSTEM_PROMPT', locals())}")
    print(f'--- USER PROMPT ---\n{user_prompt}')
    print('=' * 80 + '\n')
    raw = call_seu_llm(_get_prompt('RETRIEVAL_TERM_SYSTEM_PROMPT', locals()), user_prompt, temperature=0.0, model=SMART_MODEL_NAME, api_key=SMART_API_KEY)
    data = _safe_json_loads(raw, default=None)
    if not isinstance(data, dict):
        return {'level1': ['critical care medicine'], 'level2': ['Sepsis'], 'level3': ['diagnostic criteria']}

    def norm_list(x):
        if not isinstance(x, list):
            return []
        out = []
        for t in x:
            s = str(t).strip()
            if s:
                out.append(s)
        seen = set()
        dedup = []
        for s in out:
            if s not in seen:
                seen.add(s)
                dedup.append(s)
        return dedup
    return {'level1': norm_list(data.get('level1')), 'level2': norm_list(data.get('level2')), 'level3': norm_list(data.get('level3'))}

def build_rag_question_from_levels(levels: Dict[str, List[str]]) -> str:
    """Use spaces to connect keywords"""
    l1 = ' '.join(levels.get('level1') or [])
    l2 = ' '.join(levels.get('level2') or [])
    l3 = ' '.join(levels.get('level3') or [])
    if not (l1 or l2 or l3):
        return 'Sepsis Diagnostic Criteria Sepsis-3 SOFA Evidence of infection'
    return f'{l1} {l2} {l3}Sepsis Diagnostic Identification Standards Guide'

def format_rag_context(rag_json: Dict[str, Any], max_chars: int=6000) -> str:
    """[Fix] Compatible when data is a nested list: [[chunks], [meta], count]"""
    if not isinstance(rag_json, dict):
        return '(not recalling available knowledge: RAG returns non-dict)'
    if rag_json.get('_error'):
        return f"(RAG recall failed:{rag_json.get('_error')}）"
    data = rag_json.get('data')
    target_list = []
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], list):
            target_list = data[0]
        else:
            target_list = data
    elif isinstance(data, dict):
        if 'chunks' in data and isinstance(data['chunks'], list):
            target_list = data['chunks']
        elif 'results' in data and isinstance(data['results'], list):
            target_list = data['results']
        elif 'rows' in data and isinstance(data['rows'], list):
            target_list = data['rows']
        else:
            for v in data.values():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    if 'content' in v[0] or 'text' in v[0] or 'content_with_weight' in v[0]:
                        target_list = v
                        break
    if not target_list:
        return '(Relevant knowledge fragments were not recalled)'
    parts = []
    for i, item in enumerate(target_list[:RAG_TOP_K], start=1):
        if not isinstance(item, dict):
            continue
        text = item.get('content_with_weight') or item.get('content') or item.get('text') or item.get('chunk') or item.get('passage') or ''
        if not text and isinstance(item.get('document'), dict):
            text = item['document'].get('content') or item['document'].get('text') or ''
        score = item.get('score') or item.get('similarity') or item.get('vector_score') or ''
        meta_title = ''
        if isinstance(item.get('meta'), dict):
            meta_title = item['meta'].get('title') or item['meta'].get('source') or ''
        if not meta_title:
            meta_title = item.get('doc_name') or item.get('title') or item.get('source') or ''
        txt = str(text).strip()
        if not txt:
            continue
        head = f'[{i}]'
        if meta_title:
            head += f'source:{meta_title}'
        if score != '':
            try:
                head += f'Similarity:{float(score):.4f}'
            except:
                head += f'Similarity:{score}'
        parts.append(head + '\n' + txt)
    joined = '\n\n'.join(parts).strip()
    if not joined:
        return '(No available knowledge fragments recalled)'
    if len(joined) > max_chars:
        joined = joined[:max_chars] + '\n...(truncated)'
    return joined

def summarize_rag_context(raw_context: str) -> str:
    """The raw text recalled by RAG is summarized and cleaned using the DeepSeek model."""
    if 'No relevant knowledge fragments were recalled' in raw_context or 'Recall failed' in raw_context or (not raw_context.strip()):
        return '(The knowledge base does not provide valid reference information)'
    user_prompt = _get_prompt('experiments/public_mimic/vital_tree_rag/decide_sepsis_p.py::inline_L321', locals())
    print('\n' + '=' * 30 + ' [RAG SUMMARY PROMPT] ' + '=' * 30)
    print(f'Model: {SMART_MODEL_NAME}')
    print(f"--- SYSTEM PROMPT ---\n{_get_prompt('EVIDENCE_SYNTHESIS_SYSTEM_PROMPT', locals())}")
    print(f'--- USER PROMPT (Preview) ---\n{user_prompt[:500]}...')
    print('=' * 80 + '\n')
    summary = call_seu_llm(_get_prompt('EVIDENCE_SYNTHESIS_SYSTEM_PROMPT', locals()), user_prompt, temperature=0.1, model=SMART_MODEL_NAME, api_key=SMART_API_KEY)
    print('\n' + '#' * 30 + ' [RAG SUMMARY RESULT] ' + '#' * 30)
    print(summary)
    print('#' * 80 + '\n')
    return summary

def build_sepsis_prompt(subject_id: Any, hadm_id: Any, stay_id: Any, chart_day: Any, summary_text: str, refined_rag_context: str, keywords: Dict[str, List[str]]) -> str:
    return _get_prompt('experiments/public_mimic/vital_tree_rag/decide_sepsis_p.py::build_sepsis_prompt', locals())

def judge_sepsis_for_group_last_row(row: pd.Series) -> str:
    """For the "last line" record of a hospitalization:
    1) Use LLM (DeepSeek) to extract keywords
    2) RAGFlow recalls original clips
    3) Use LLM (DeepSeek) to summarize the original fragment -> get the Refined Context
    4) Use LLM (Qwen) for final judgment"""
    summary_text = str(row.get('icu_progress_summary', '') or '').strip()
    chart_day = row.get('chart_day', '')
    subject_id = row.get('subject_id', '')
    hadm_id = row.get('hadm_id', '')
    stay_id = row.get('stay_id', '')
    if not summary_text:
        return 'The icu_progress_summary text is missing and the model reply cannot be generated.'
    keywords = extract_3level_keywords(summary_text)
    rag_question = build_rag_question_from_levels(keywords)
    print(f'\n >>> RAG search terms:{rag_question}')
    rag_json = ragflow_retrieval(rag_question)
    raw_rag_context = format_rag_context(rag_json)
    refined_rag_context = summarize_rag_context(raw_rag_context)
    user_prompt = build_sepsis_prompt(subject_id=subject_id, hadm_id=hadm_id, stay_id=stay_id, chart_day=chart_day, summary_text=summary_text, refined_rag_context=refined_rag_context, keywords=keywords)
    print('\n' + '=' * 30 + ' [Sepsis JUDGE PROMPT] ' + '=' * 30)
    print(f'Model: {MODEL_NAME} (Default)')
    print(f"--- SYSTEM PROMPT ---\n{_get_prompt('PUBLIC_CLASSIFICATION_SYSTEM_PROMPT', locals())}")
    print(f'--- USER PROMPT ---\n{user_prompt}')
    print('=' * 80 + '\n')
    raw = call_seu_llm(_get_prompt('PUBLIC_CLASSIFICATION_SYSTEM_PROMPT', locals()), user_prompt, temperature=0.0)
    return str(raw).strip()

def main():
    start_time = time.time()
    if os.path.exists(CHECKPOINT_FILE):
        print(f'>>> Found the breakpoint file and will continue from the breakpoint:{CHECKPOINT_FILE}')
        df = pd.read_excel(CHECKPOINT_FILE)
    else:
        print(f'>>> Reading input file:{INPUT_FILE}')
        df = pd.read_excel(INPUT_FILE)
    if 'icu_progress_summary' not in df.columns:
        raise RuntimeError("The 'icu_progress_summary' column was not found in the input table, please make sure you are using a results table with daily summary.")
    if 'chart_day' in df.columns:
        try:
            df['chart_day'] = pd.to_datetime(df['chart_day'])
        except Exception:
            pass
    OUT_COL = 'sepsis_reasoning'
    if OUT_COL not in df.columns:
        df[OUT_COL] = ''
    group_cols = [c for c in ['subject_id', 'hadm_id', 'stay_id'] if c in df.columns]
    if group_cols:
        grouped_obj = df.groupby(group_cols, dropna=False)
    else:
        grouped_obj = [('single', df)]
    tasks = []
    group_indices: Dict[Any, pd.Index] = {}
    for key, sub_df in grouped_obj:
        key_tuple = (key,) if isinstance(key, str) else key if isinstance(key, tuple) else (key,)
        if sub_df.empty:
            continue
        if 'chart_day' in sub_df.columns:
            sub_sorted = sub_df.sort_values('chart_day')
        else:
            sub_sorted = sub_df.sort_index()
        last_idx = sub_sorted.index[-1]
        group_indices[key_tuple] = sub_df.index
        tasks.append((key_tuple, last_idx))
    total_groups = len(tasks)
    print(f'>>> Detected in total{total_groups}A hospitalization requires Sepsis+RAG analysis.')

    def group_done(key_tuple) -> bool:
        idxs = group_indices.get(key_tuple, [])
        if len(idxs) == 0:
            return False
        vals = df.loc[idxs, OUT_COL]
        ok = True
        for v in vals:
            s = str(v).strip() if v is not None else ''
            if not s or s.startswith('<<LLM_ERROR>>'):
                ok = False
                break
        return ok
    pending_tasks = []
    for t in tasks:
        key_tuple, last_idx = t
        if group_done(key_tuple):
            continue
        pending_tasks.append(t)
    tasks = pending_tasks
    total_groups = len(tasks)
    print(f'>>> After breakpoint filtering, processing still needs to be done{total_groups}a hospitalization process.')
    if total_groups == 0:
        print('>>> All completed, no need to run.')
        df.to_excel(OUTPUT_FILE, index=False)
        return
    lock = threading.Lock()
    groups_done = 0
    groups_written_since_ckpt = 0
    results: Dict[Any, str] = {}

    def checkpoint_save():
        try:
            df.to_excel(CHECKPOINT_FILE, index=False)
            print(f'>>> [checkpoint] Saved:{CHECKPOINT_FILE}')
        except Exception as e:
            print(f'>>> [checkpoint] Save failed:{e}')

    def worker(task):
        nonlocal groups_done, groups_written_since_ckpt
        key, last_idx = task
        row = df.loc[last_idx]
        key_str = ', '.join((str(k) for k in key))
        reasoning = judge_sepsis_for_group_last_row(row)
        with lock:
            groups_done += 1
            groups_written_since_ckpt += 1
            print(f'[Sepsis+RAG Analysis Progress] Completed{groups_done}/{total_groups}hospitalization process (key={key_str})')
        return (key, reasoning)
    max_workers = min(8, total_groups)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(worker, t): t for t in tasks}
        for future in as_completed(future_to_task):
            try:
                key, reasoning = future.result()
                results[key] = reasoning
                idxs = group_indices.get(key, [])
                df.loc[idxs, OUT_COL] = reasoning
                with lock:
                    if groups_written_since_ckpt >= CHECKPOINT_EVERY_N_GROUPS:
                        groups_written_since_ckpt = 0
                        checkpoint_save()
            except Exception as e:
                print(f'[Thread task exception]{e}')
    df.to_excel(OUTPUT_FILE, index=False)
    try:
        df.to_excel(CHECKPOINT_FILE, index=False)
    except Exception:
        pass
    elapsed = time.time() - start_time
    print(f'>>> A file containing the output of the Sepsis+RAG analysis has been generated:{OUTPUT_FILE}')
    print(f'>>> Breakpoint file:{CHECKPOINT_FILE}')
    print(f'>>> Total program running time:{elapsed:.2f}Second')
if __name__ == '__main__':
    main()
