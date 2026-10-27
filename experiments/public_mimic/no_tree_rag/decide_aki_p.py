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
INPUT_FILE = 'non_aki_info_with_summary.xlsx'
OUTPUT_FILE = 'non_aki_info_with_summary_decide.xlsx'
BASE_URL = os.getenv('OPENAI_BASE_URL', 'http://localhost:8000/v1')
MODEL_NAME = 'qwen3-235b-thinking-2507-bigdata'
SEU_API_KEY = os.getenv('SEU_API_KEY', os.getenv('OPENAI_API_KEY', ''))
if not SEU_API_KEY:
    raise RuntimeError('SEU API Key not found, please set SEU_API_KEY or fill it in the script.')
MAX_RETRIES = 6
RETRY_BASE_SLEEP = 2.0
RETRY_MAX_SLEEP = 60.0
CHECKPOINT_EVERY_N_GROUPS = 5
CHECKPOINT_FILE = OUTPUT_FILE.replace('.xlsx', '.checkpoint.xlsx')
SEU_SESSION = requests.Session()
SEU_SESSION.trust_env = False

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

def call_seu_llm(system_prompt: str, user_prompt: str, temperature: float=0.0) -> str:
    """Call the SEU large model interface (with retry) and return text results."""
    url = f'{BASE_URL}/chat/completions'
    headers = {'Authorization': f'Bearer {SEU_API_KEY}', 'Content-Type': 'application/json'}
    payload = {'model': MODEL_NAME, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}], 'temperature': temperature}
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
            print(f'[SEU LLM retry] attempt={attempt}/{MAX_RETRIES}, error={e}, sleep={sleep_s:.1f}s')
            time.sleep(sleep_s)
    return f'<<LLM_ERROR>> {str(last_err)}'

def build_aki_prompt(subject_id: Any, hadm_id: Any, stay_id: Any, chart_day: Any, summary_text: str) -> str:
    return _get_prompt('experiments/public_mimic/no_tree_rag/decide_aki_p.py::build_aki_prompt', locals())

def judge_aki_for_group_last_row(row: pd.Series) -> str:
    """For the "last line" record of a hospitalization:
    1) Read icu_progress_summary directly
    2) No RAG recall
    3) LLM directly outputs natural language responses"""
    summary_text = str(row.get('icu_progress_summary', '') or '').strip()
    chart_day = row.get('chart_day', '')
    subject_id = row.get('subject_id', '')
    hadm_id = row.get('hadm_id', '')
    stay_id = row.get('stay_id', '')
    if not summary_text:
        return 'The icu_progress_summary text is missing and the model reply cannot be generated.'
    user_prompt = build_aki_prompt(subject_id=subject_id, hadm_id=hadm_id, stay_id=stay_id, chart_day=chart_day, summary_text=summary_text)
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
    OUT_COL = 'aki_reasoning'
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
    print(f'>>> Detected in total{total_groups}A hospitalization requires AKI analysis (No RAG).')

    def group_done(key_tuple) -> bool:
        idxs = group_indices.get(key_tuple, [])
        if len(idxs) == 0:
            return False
        vals = df.loc[idxs, OUT_COL]
        ok = True
        for v in vals:
            s = str(v).strip() if v is not None else ''
            if not s:
                ok = False
                break
            if s.startswith('<<LLM_ERROR>>'):
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
        reasoning = judge_aki_for_group_last_row(row)
        with lock:
            groups_done += 1
            groups_written_since_ckpt += 1
            print(f'[AKI Analysis Progress] Completed{groups_done}/{total_groups}hospitalization process (key={key_str})')
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
    print(f'>>> File containing AKI analysis output has been generated:{OUTPUT_FILE}')
    print(f'>>> Breakpoint file:{CHECKPOINT_FILE}')
    print(f'>>> Total program running time:{elapsed:.2f}Second')
if __name__ == '__main__':
    main()
