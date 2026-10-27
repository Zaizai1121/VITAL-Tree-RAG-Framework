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
from typing import Optional, Tuple
import pandas as pd
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
INPUT_FILE = 'sepsis_info.xlsx'
OUTPUT_FILE = 'sepsis_info_with_summary.xlsx'
BASE_URL = os.getenv('OPENAI_BASE_URL', 'http://localhost:8000/v1')
MODEL_NAME = 'qwen3-235b-thinking-2507'
API_KEY = os.getenv('SEU_API_KEY', os.getenv('OPENAI_API_KEY', ''))
if not API_KEY:
    raise RuntimeError('API Key not found, please set the environment variable SEU_API_KEY or fill in API_KEY in the script.')

def safe_get(row: pd.Series, col: Optional[str]) -> str:
    """Safely convert NaN/missing to string, avoiding KeyError."""
    if not col:
        return ''
    if col not in row:
        return ''
    val = row[col]
    if pd.isna(val):
        return ''
    return str(val)
LLM_COLS = ['subject_id', 'hadm_id', 'stay_id', 'icu_intime', 'sofa_time', 'sofa_score', 'sofa_respiration', 'sofa_coagulation', 'sofa_liver', 'sofa_cardiovascular', 'sofa_cns', 'sofa_renal', 'chartevents_json', 'labevents_json', 'inputevents_json', 'outputevents_json', 'procedureevents_json', 'microbiology_json', 'prescriptions_json', 'pharmacy_json', 'emar_json', 'procedures_text_json']

def get_window_fields(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """New table: half-day branches are usually day_start + day_end
    Old table: maybe only chart_day
    Here is the minimum compatibility:
      - priority (day_start, day_end)
      - Degenerates into (chart_day, None)"""
    if 'day_start' in df.columns:
        end_field = 'day_end' if 'day_end' in df.columns else None
        return ('day_start', end_field)
    if 'chart_day' in df.columns:
        return ('chart_day', None)
    return (None, None)

def build_first_prompt(row: pd.Series, start_field: Optional[str], end_field: Optional[str]) -> str:
    return _get_prompt('experiments/public_mimic/vital_tree_rag/summary_sepsis.py::build_first_prompt', locals())

def build_update_prompt(prev_summary: str, prev_window: str, row: pd.Series, start_field: Optional[str], end_field: Optional[str]) -> str:
    return _get_prompt('experiments/public_mimic/vital_tree_rag/summary_sepsis.py::build_update_prompt', locals())
MAX_RETRIES = 6
RETRY_BASE_SLEEP = 2.0
RETRY_MAX_SLEEP = 60.0
CHECKPOINT_EVERY_N_ROWS = 10
CHECKPOINT_FILE = OUTPUT_FILE.replace('.xlsx', '.checkpoint.xlsx')

def _safe_parse_json_response(resp: requests.Response):
    """Try to parse the response into a dict:
    - JSON parsing failed -> return {"_raw_text": ..., "_status_code": ...}
    - Empty response / None -> Return {"_raw_text": None, ...}"""
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

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the SEU large model interface to return the generated summary text. (Enhancement: retry + robust parsing to avoid direct explosion of None)"""
    url = f'{BASE_URL}/chat/completions'
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    payload = {'model': MODEL_NAME, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}], 'temperature': 0.2}
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=500)
            if not 200 <= resp.status_code < 300:
                data = _safe_parse_json_response(resp)
                raise RuntimeError(f'HTTP {resp.status_code}, body={data}')
            data = _safe_parse_json_response(resp)
            if isinstance(data, dict) and 'choices' in data and data['choices']:
                msg = data['choices'][0].get('message') if isinstance(data['choices'][0], dict) else None
                if isinstance(msg, dict) and 'content' in msg and (msg['content'] is not None):
                    return str(msg['content']).strip()
            if isinstance(data, dict) and 'choices' in data and data['choices']:
                c0 = data['choices'][0]
                if isinstance(c0, dict) and 'text' in c0 and (c0['text'] is not None):
                    return str(c0['text']).strip()
            raise RuntimeError(f'The return structure is not as expected:{data}')
        except Exception as e:
            last_err = e
            sleep_s = min(RETRY_MAX_SLEEP, RETRY_BASE_SLEEP * 2 ** (attempt - 1))
            print(f'[LLM retry] attempt={attempt}/{MAX_RETRIES}, error={e}, sleep={sleep_s:.1f}s')
            time.sleep(sleep_s)
    return f'<<LLM_ERROR>> {str(last_err)}'

def main():
    start_time = time.time()
    if os.path.exists(CHECKPOINT_FILE):
        print(f'>>> Found the breakpoint file and will continue from the breakpoint:{CHECKPOINT_FILE}')
        df = pd.read_excel(CHECKPOINT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)
    start_field, end_field = get_window_fields(df)
    if start_field and start_field in df.columns:
        try:
            df[start_field] = pd.to_datetime(df[start_field])
        except Exception:
            pass
    if end_field and end_field in df.columns:
        try:
            df[end_field] = pd.to_datetime(df[end_field])
        except Exception:
            pass
    if 'icu_progress_summary' not in df.columns:
        df['icu_progress_summary'] = ''
    group_cols = [c for c in ['subject_id', 'hadm_id', 'stay_id'] if c in df.columns]
    if group_cols:
        grouped_obj = df.groupby(group_cols)
        groups = [(key, sub_df.copy()) for key, sub_df in grouped_obj]
    else:
        groups = [(('single',), df.copy())]
    total_groups = len(groups)
    total_rows = len(df)
    if total_groups == 0 or total_rows == 0:
        print('The input table is empty and the program ends.')
        return
    lock = threading.Lock()
    groups_done = 0
    rows_done = 0
    rows_written_since_ckpt = 0

    def _should_skip_row(idx) -> bool:
        """If the line already has a summary (not empty and not an error placeholder), it will be skipped, and breakpoint continuation is supported."""
        try:
            v = df.loc[idx, 'icu_progress_summary']
        except Exception:
            return False
        if v is None:
            return False
        s = str(v).strip()
        if not s:
            return False
        if s.startswith('<<LLM_ERROR>>'):
            return False
        return True

    def _checkpoint_save():
        """Move the current df to the checkpoint to avoid crashing and re-running."""
        try:
            df.to_excel(CHECKPOINT_FILE, index=False)
            print(f'>>> [checkpoint] Saved:{CHECKPOINT_FILE}')
        except Exception as e:
            print(f'>>> [checkpoint] Save failed:{e}')

    def process_group(args):
        nonlocal groups_done, rows_done, rows_written_since_ckpt
        key, sub_df = args
        if start_field and start_field in sub_df.columns:
            sub_df = sub_df.sort_values(start_field)
        else:
            sub_df = sub_df.sort_index()
        prev_summary: Optional[str] = None
        prev_window: Optional[str] = None
        summaries = []
        key_str = ', '.join((str(k) for k in (key if isinstance(key, tuple) else (key,))))
        for idx, row in sub_df.iterrows():
            if _should_skip_row(idx):
                with lock:
                    rows_done += 1
                    print(f'[row progress][skip completed]{rows_done}/{total_rows}(Current group:{key_str})')
                prev_summary = str(df.loc[idx, 'icu_progress_summary'])
                cur_start = safe_get(row, start_field)
                cur_end = safe_get(row, end_field)
                prev_window = f'{cur_start} ~ {cur_end}' if cur_start and cur_end else cur_start
                continue
            cur_start = safe_get(row, start_field)
            cur_end = safe_get(row, end_field)
            cur_window = f'{cur_start} ~ {cur_end}' if cur_start and cur_end else cur_start
            if prev_summary is None:
                user_prompt = build_first_prompt(row, start_field, end_field)
            else:
                prev_window_str = str(prev_window) if prev_window is not None else ''
                user_prompt = build_update_prompt(prev_summary, prev_window_str, row, start_field, end_field)
            summary_text = call_llm(_get_prompt('PUBLIC_LONGITUDINAL_SUMMARY_SYSTEM_PROMPT', locals()), user_prompt)
            summaries.append((idx, summary_text))
            prev_summary = summary_text
            prev_window = cur_window
            with lock:
                rows_done += 1
                rows_written_since_ckpt += 1
                print(f'[row progress]{rows_done}/{total_rows} (Current group:{key_str})')
                if rows_written_since_ckpt >= CHECKPOINT_EVERY_N_ROWS:
                    rows_written_since_ckpt = 0
                    _checkpoint_save()
        with lock:
            groups_done += 1
            print(f'[Group Progress] Completed{groups_done}/{total_groups}hospitalization process (key={key_str})')
        return summaries
    all_summaries = []
    max_workers = min(8, total_groups)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_group, g) for g in groups]
        for future in as_completed(futures):
            try:
                res = future.result()
                all_summaries.extend(res)
            except Exception as e:
                print(f'[Thread task exception]{e}')
    for idx, text in all_summaries:
        df.loc[idx, 'icu_progress_summary'] = text
    df.to_excel(OUTPUT_FILE, index=False)
    try:
        df.to_excel(CHECKPOINT_FILE, index=False)
    except Exception:
        pass
    elapsed = time.time() - start_time
    print(f'File generated with day by day summary:{OUTPUT_FILE}')
    print(f'Total program running time:{elapsed:.2f}Second')
if __name__ == '__main__':
    main()
