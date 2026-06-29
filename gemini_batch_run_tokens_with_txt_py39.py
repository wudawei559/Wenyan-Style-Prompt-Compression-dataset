import csv
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, List, Dict

from google import genai
from google.genai import types

# =========================
# 你主要改这里
# =========================
MODEL_NAME = "gemini-2.5-flash"
# 每个 prompt 目标成功多少次
SUCCESSFUL_RUNS_PER_PROMPT = 3

# 每个 prompt 最多总尝试多少次（含失败）
MAX_TOTAL_ATTEMPTS_PER_PROMPT = 20

THINKING_BUDGET = 0  # 为了省 token，建议保持 0

# 这两个间隔主要是为了降低免费额度下的 429 / RESOURCE_EXHAUSTED 概率
SLEEP_BETWEEN_ATTEMPTS = 5.0   # 每次尝试之间停一下
SLEEP_BETWEEN_PROMPTS = 5.0    # 每个 prompt 完成后额外停一下

# 单次尝试内部重试
MAX_RETRIES_PER_ATTEMPT = 5
RETRY_BACKOFF_SECONDS = 8.0

# 是否在每次 generate 前调用 count_tokens
# 正式统计时建议优先使用 response_* 列；关闭这里可以少一次 API 请求，更稳
USE_PRECOUNT = False

OUTPUT_DIR = Path("gemini_experiment_outputs")
RUN_LABEL = "batch_run"
SAVE_TXT_BACKUP = True

# 你只需要在这里粘贴/修改 prompt。
# 支持两种写法：
# 1) 直接写字符串
# 2) 写成 {"prompt_id": "P01", "text": "..."}
PROMPTS: List[Any] = [
    {
        "prompt_id": "P15_original",
        "text": "请比较“写得尽量简短”和“写得清楚完整”这两种prompt写作原则可能带来的差异；如果目标是既节省token又避免误解，应如何权衡？",
    },
    {
        "prompt_id": "P15_free_compressed",
        "text": "比较“尽量简短”和“清楚完整”这两种prompt写作原则的差异；若目标是既省token又避免误解，应如何权衡？",
    },
    {
        "prompt_id": "P15_classical_compressed",
        "text": "请比较“务求简短”与“务求明备”两种prompt写法之异；若欲既省token又免歧解，当如何权衡？",
    },
]


@dataclass
class PromptItem:
    prompt_id: str
    text: str


def get_api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def build_client() -> genai.Client:
    api_key = get_api_key()
    if not api_key:
        print("错误：未检测到环境变量 GEMINI_API_KEY 或 GOOGLE_API_KEY。")
        print("请先在 PowerShell 中执行：")
        print('  setx GEMINI_API_KEY "你的_API_Key"')
        print("然后关闭并重新打开 PowerShell。")
        raise SystemExit(1)
    return genai.Client(api_key=api_key)


def normalize_prompts(raw_prompts: Iterable[Any]) -> List[PromptItem]:
    normalized: List[PromptItem] = []
    for idx, item in enumerate(raw_prompts, start=1):
        if isinstance(item, str):
            normalized.append(PromptItem(prompt_id="P{0:02d}".format(idx), text=item.strip()))
            continue

        if isinstance(item, dict):
            prompt_id = str(item.get("prompt_id") or "P{0:02d}".format(idx)).strip()
            text = str(item.get("text") or "").strip()
            if not text:
                raise ValueError("第 {0} 个 prompt 缺少 text。".format(idx))
            normalized.append(PromptItem(prompt_id=prompt_id, text=text))
            continue

        raise TypeError(
            "第 {0} 个 prompt 格式不支持。请使用字符串，或 {{'prompt_id': 'P01', 'text': '...'}}。".format(idx)
        )

    if not normalized:
        raise ValueError("PROMPTS 为空，请先填入至少一个 prompt。")
    return normalized


def build_generation_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
    )


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def extract_text(response: Any) -> str:
    text = safe_getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts: List[str] = []
    for cand in safe_getattr(response, "candidates", []) or []:
        content = safe_getattr(cand, "content", None)
        for part in safe_getattr(content, "parts", []) or []:
            part_text = safe_getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                parts.append(part_text)
    return "\n".join(parts).strip()


def usage_to_dict(usage: Any) -> Dict[str, Any]:
    return {
        "response_prompt_token_count": safe_getattr(usage, "prompt_token_count", None),
        "response_output_token_count": safe_getattr(usage, "candidates_token_count", None),
        "response_total_token_count": safe_getattr(usage, "total_token_count", None),
        "response_thoughts_token_count": safe_getattr(usage, "thoughts_token_count", None),
        "response_cached_content_token_count": safe_getattr(
            usage, "cached_content_token_count", None
        ),
    }


def count_input_tokens(client: genai.Client, model_name: str, prompt_text: str) -> Any:
    count_resp = client.models.count_tokens(model=model_name, contents=prompt_text)
    return safe_getattr(count_resp, "total_tokens", None)


def run_single_request(
    client: genai.Client,
    model_name: str,
    prompt_id: str,
    prompt_text: str,
    successful_run_index: int,
    global_attempt_index: int,
    config: types.GenerateContentConfig,
) -> Dict[str, Any]:
    last_error: Optional[str] = None

    for retry_index in range(1, MAX_RETRIES_PER_ATTEMPT + 1):
        started_at = datetime.now().isoformat(timespec="seconds")
        try:
            input_tokens_precount = None
            if USE_PRECOUNT:
                input_tokens_precount = count_input_tokens(client, model_name, prompt_text)

            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=config,
            )

            usage = usage_to_dict(safe_getattr(response, "usage_metadata", None))
            output_text = extract_text(response)

            return {
                "prompt_id": prompt_id,
                "run_index": successful_run_index,            # 第几个成功结果
                "global_attempt_index": global_attempt_index, # 这是第几次总尝试
                "retry_index": retry_index,                   # 当前总尝试内部第几次重试
                "status": "ok",
                "started_at": started_at,
                "model_name": model_name,
                "prompt_text": prompt_text,
                "input_tokens_precount": input_tokens_precount,
                **usage,
                "output_text": output_text,
                "error": "",
            }
        except Exception as exc:
            last_error = repr(exc)
            if retry_index < MAX_RETRIES_PER_ATTEMPT:
                time.sleep(RETRY_BACKOFF_SECONDS * retry_index)
            else:
                return {
                    "prompt_id": prompt_id,
                    "run_index": successful_run_index,
                    "global_attempt_index": global_attempt_index,
                    "retry_index": retry_index,
                    "status": "error",
                    "started_at": started_at,
                    "model_name": model_name,
                    "prompt_text": prompt_text,
                    "input_tokens_precount": None,
                    "response_prompt_token_count": None,
                    "response_output_token_count": None,
                    "response_total_token_count": None,
                    "response_thoughts_token_count": None,
                    "response_cached_content_token_count": None,
                    "output_text": "",
                    "error": last_error,
                }

    raise RuntimeError("不应运行到这里。")


def build_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault(str(row["prompt_id"]), []).append(row)

    def mean_of(key: str, items: List[Dict[str, Any]]) -> Optional[float]:
        vals = [item[key] for item in items if isinstance(item.get(key), (int, float))]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 4)

    summary: List[Dict[str, Any]] = []
    for prompt_id, items in grouped.items():
        summary.append(
            {
                "prompt_id": prompt_id,
                "successful_runs": len(items),
                "mean_input_tokens_precount": mean_of("input_tokens_precount", items),
                "mean_response_prompt_token_count": mean_of(
                    "response_prompt_token_count", items
                ),
                "mean_response_output_token_count": mean_of(
                    "response_output_token_count", items
                ),
                "mean_response_total_token_count": mean_of(
                    "response_total_token_count", items
                ),
                "mean_response_thoughts_token_count": mean_of(
                    "response_thoughts_token_count", items
                ),
                "error_rows": sum(1 for item in rows if item.get("prompt_id") == prompt_id and item.get("status") != "ok"),
            }
        )

    summary.sort(key=lambda x: x["prompt_id"])
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_excel(run_rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]], path: Path) -> str:
    try:
        import pandas as pd
    except Exception:
        return "未生成 xlsx（当前环境未安装 pandas）。CSV 已生成。"

    run_df = pd.DataFrame(run_rows)
    summary_df = pd.DataFrame(summary_rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        run_df.to_excel(writer, sheet_name="run_level_results", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
    return "xlsx 已生成。"


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned[:120] or "untitled"


def write_txt_backup(txt_dir: Path, row: Dict[str, Any]) -> Path:
    file_name = "{0}_attempt{1:02d}_{2}.txt".format(
        sanitize_filename(str(row["prompt_id"])),
        int(row.get("global_attempt_index", 0)),
        str(row.get("status", "unknown")),
    )
    txt_path = txt_dir / file_name

    lines = [
        "prompt_id: {0}".format(row.get("prompt_id", "")),
        "run_index: {0}".format(row.get("run_index", "")),
        "global_attempt_index: {0}".format(row.get("global_attempt_index", "")),
        "retry_index: {0}".format(row.get("retry_index", "")),
        "status: {0}".format(row.get("status", "")),
        "started_at: {0}".format(row.get("started_at", "")),
        "model_name: {0}".format(row.get("model_name", "")),
        "input_tokens_precount: {0}".format(row.get("input_tokens_precount", "")),
        "response_prompt_token_count: {0}".format(row.get("response_prompt_token_count", "")),
        "response_output_token_count: {0}".format(row.get("response_output_token_count", "")),
        "response_total_token_count: {0}".format(row.get("response_total_token_count", "")),
        "response_thoughts_token_count: {0}".format(row.get("response_thoughts_token_count", "")),
        "response_cached_content_token_count: {0}".format(row.get("response_cached_content_token_count", "")),
        "error: {0}".format(row.get("error", "")),
        "",
        "=== PROMPT ===",
        str(row.get("prompt_text", "")),
        "",
        "=== OUTPUT ===",
        str(row.get("output_text", "")),
        "",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return txt_path


def main() -> int:
    prompts = normalize_prompts(PROMPTS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / "{0}_{1}".format(RUN_LABEL, timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_csv_path = run_dir / "{0}_{1}_run_level_results.csv".format(RUN_LABEL, timestamp)
    summary_csv_path = run_dir / "{0}_{1}_summary.csv".format(RUN_LABEL, timestamp)
    run_xlsx_path = run_dir / "{0}_{1}_results.xlsx".format(RUN_LABEL, timestamp)
    txt_dir = run_dir / "txt_backups"
    if SAVE_TXT_BACKUP:
        txt_dir.mkdir(parents=True, exist_ok=True)

    client = build_client()
    config = build_generation_config()
    all_rows: List[Dict[str, Any]] = []

    try:
        print("开始运行：{0} 个 prompt；每个目标成功 {1} 次。".format(len(prompts), SUCCESSFUL_RUNS_PER_PROMPT))
        print("模型：{0}".format(MODEL_NAME))
        print("提示：本脚本使用的是单轮 generate_content，而不是 chat，因此 token 不会跨轮累加。")
        print("提示：当前 USE_PRECOUNT = {0}。正式统计建议优先使用 response_* 列。".format(USE_PRECOUNT))
        print()

        for prompt_idx, prompt in enumerate(prompts, start=1):
            success_count = 0
            attempt_count = 0
            print("===== 开始 {0}/{1}: {2} =====".format(prompt_idx, len(prompts), prompt.prompt_id))

            while success_count < SUCCESSFUL_RUNS_PER_PROMPT and attempt_count < MAX_TOTAL_ATTEMPTS_PER_PROMPT:
                attempt_count += 1
                target_run_index = success_count + 1
                print(
                    "[{0}] 目标成功第 {1}/{2} 次；当前总尝试 {3}/{4} ...".format(
                        prompt.prompt_id,
                        target_run_index,
                        SUCCESSFUL_RUNS_PER_PROMPT,
                        attempt_count,
                        MAX_TOTAL_ATTEMPTS_PER_PROMPT,
                    )
                )

                row = run_single_request(
                    client=client,
                    model_name=MODEL_NAME,
                    prompt_id=prompt.prompt_id,
                    prompt_text=prompt.text,
                    successful_run_index=target_run_index,
                    global_attempt_index=attempt_count,
                    config=config,
                )
                all_rows.append(row)
                write_csv(run_csv_path, all_rows)
                if SAVE_TXT_BACKUP:
                    write_txt_backup(txt_dir, row)

                if row.get("status") == "ok":
                    success_count += 1
                    print("  -> 成功。已拿到 {0}/{1} 个成功结果。".format(success_count, SUCCESSFUL_RUNS_PER_PROMPT))
                else:
                    print("  -> 失败：{0}".format(row.get("error", "")))

                if success_count < SUCCESSFUL_RUNS_PER_PROMPT and attempt_count < MAX_TOTAL_ATTEMPTS_PER_PROMPT:
                    if SLEEP_BETWEEN_ATTEMPTS > 0:
                        time.sleep(SLEEP_BETWEEN_ATTEMPTS)

            print(
                "===== {0} 完成：成功 {1}/{2}，总尝试 {3} 次 =====".format(
                    prompt.prompt_id, success_count, SUCCESSFUL_RUNS_PER_PROMPT, attempt_count
                )
            )
            if prompt_idx < len(prompts) and SLEEP_BETWEEN_PROMPTS > 0:
                time.sleep(SLEEP_BETWEEN_PROMPTS)

    finally:
        try:
            client.close()
        except Exception:
            pass

    summary_rows = build_summary_rows(all_rows)
    write_csv(summary_csv_path, summary_rows)
    excel_message = maybe_write_excel(all_rows, summary_rows, run_xlsx_path)

    print("\n=== 完成 ===")
    print("结果目录：{0}".format(run_dir.resolve()))
    print("逐次结果 CSV：{0}".format(run_csv_path.resolve()))
    print("汇总结果 CSV：{0}".format(summary_csv_path.resolve()))
    print("Excel：{0}  （{1}）".format(run_xlsx_path.resolve(), excel_message))
    if SAVE_TXT_BACKUP:
        print("TXT 备份目录：{0}".format(txt_dir.resolve()))
    print("\n你最关心的点：")
    print("1) 每一行都是‘单次尝试’的 token；成功与失败都会保留。")
    print("2) 只要你不改成 chat，会话历史就不会自动带入下一轮。")
    print("3) 默认关闭了 count_tokens，减少一次 API 调用，更不容易被限流。")
    print("4) 正式统计时建议优先用 response_prompt_token_count / response_output_token_count / response_total_token_count。")
    print("5) 如果某个 prompt 最终没拿满 3 次成功，请先看 run_level_results.csv 里的 error 列。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
