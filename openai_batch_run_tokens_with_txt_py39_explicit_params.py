import csv
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, List, Dict

from openai import OpenAI

# =========================
# 你主要改这里
# =========================
MODEL_NAME = "gpt-5.4"

# 每个 prompt 目标成功多少次
SUCCESSFUL_RUNS_PER_PROMPT = 3

# 每个 prompt 最多总尝试多少次（含失败）
MAX_TOTAL_ATTEMPTS_PER_PROMPT = 20

# 这两个间隔主要是为了降低偶发报错概率
SLEEP_BETWEEN_ATTEMPTS = 3.0
SLEEP_BETWEEN_PROMPTS = 3.0

# 单次尝试内部重试
MAX_RETRIES_PER_ATTEMPT = 5
RETRY_BACKOFF_SECONDS = 6.0

OUTPUT_DIR = Path("openai_experiment_outputs")
RUN_LABEL = "batch_run"
SAVE_TXT_BACKUP = True

# 若想显式控制思考深度，可改成 "none" / "low" / "medium" / "high" / "xhigh"
# 你当前按要求保持默认即可
REASONING_EFFORT: Optional[str] = None

# 显式固定这三个参数，便于论文方法部分复现
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_OUTPUT_TOKENS = 65536

# 你只需要在这里粘贴/修改 prompt。
# 支持两种写法：
# 1) 直接写字符串
# 2) 写成 {"prompt_id": "P01", "text": "..."}
PROMPTS: List[Any] = [
    {
        "prompt_id": "P01_original",
        "text": "请帮我写一封发给导师的邮件，说明我已经完成论文初稿修改，希望下周找一个方便的时间讨论，语气礼貌自然，不要显得太生硬。",
    },
    {
        "prompt_id": "P01_free_compressed",
        "text": "写一封给导师的邮件，说明我已完成论文初稿修改，希望下周约个方便时间讨论，语气礼貌自然，不要太生硬。",
    },
    {
        "prompt_id": "P01_classical_compressed",
        "text": "请拟致导师邮件一封，言余已毕论文初稿修订，愿于下周择便商讨，语气须恭谨自然，毋过生硬。",
    },
    {
        "prompt_id": "P02_original",
        "text": "请围绕“人工智能如何影响大学生写作”这个主题列一个三部分的论文提纲，每部分用一句话说明重点，整体逻辑要清楚。",
    },
    {
        "prompt_id": "P02_free_compressed",
        "text": "围绕“人工智能如何影响大学生写作”列一个三部分论文提纲，每部分用一句话说明重点，逻辑清楚。",
    },
    {
        "prompt_id": "P02_classical_compressed",
        "text": "请就“人工智能如何影响大学生写作”拟论文纲目三端，各以一句明其重点，脉络须清。",
    },
    {
        "prompt_id": "P03_original",
        "text": "请比较线上开会和线下开会的优缺点；如果我的目标是提高沟通效率，应优先选择哪一种，并说明理由。",
    },
    {
        "prompt_id": "P03_free_compressed",
        "text": "比较线上开会和线下开会的优缺点；若目标是提高沟通效率，应优先选择哪一种，并说明理由。",
    },
    {
        "prompt_id": "P03_classical_compressed",
        "text": "请比较线上与线下会议之利弊；若以提高沟通效率为要，当先取何者，并申其故。",
    },
    {
        "prompt_id": "P04_original",
        "text": "请把“我这几天事情很多，可能没有办法按时完成，但我会尽量推进”改写得更委婉、更正式一些。",
    },
    {
        "prompt_id": "P04_free_compressed",
        "text": "把“我这几天事情很多，可能没有办法按时完成，但我会尽量推进”改得更委婉、正式一些。",
    },
    {
        "prompt_id": "P04_classical_compressed",
        "text": "请将“我这几天事情很多，可能没有办法按时完成，但我会尽量推进”改为较婉且正式之辞。",
    },
    {
        "prompt_id": "P05_original",
        "text": "请帮我生成一段适合学术场合使用的自我介绍，要求简洁、真诚，不要夸张，长度控制在120字以内。",
    },
    {
        "prompt_id": "P05_free_compressed",
        "text": "生成一段适合学术场合的自我介绍，要求简洁、真诚、不夸张，控制在120字以内。",
    },
    {
        "prompt_id": "P05_classical_compressed",
        "text": "请拟一段宜于学术场合之自述，务求简洁诚恳，毋事夸饰，限一百二十字内。",
    },
    {
        "prompt_id": "P06_original",
        "text": "请为“中文用户使用大模型时如何节省token成本”这一主题拟一个论文标题，并提出三个研究问题。",
    },
    {
        "prompt_id": "P06_free_compressed",
        "text": "为“中文用户使用大模型时如何节省token成本”拟一个论文标题，并提出三个研究问题。",
    },
    {
        "prompt_id": "P06_classical_compressed",
        "text": "请为“中文用户使用大模型时如何节省token成本”拟论文题目一则，并列研究问题三端。",
    },
    {
        "prompt_id": "P07_original",
        "text": "请把下面这句话翻译成自然英文：我想研究一种更适合中文用户的大模型提示压缩方式。",
    },
    {
        "prompt_id": "P07_free_compressed",
        "text": "把这句话翻译成自然英文：我想研究一种更适合中文用户的大模型提示压缩方式。",
    },
    {
        "prompt_id": "P07_classical_compressed",
        "text": "请将此句译为自然英文：我想研究一种更适合中文用户的大模型提示压缩方式。",
    },
    {
        "prompt_id": "P08_original",
        "text": "请把“这项研究很有意义，因为它不仅节省成本，也提高了交互效率”改写得更像论文里的语言。",
    },
    {
        "prompt_id": "P08_free_compressed",
        "text": "把“这项研究很有意义，因为它不仅节省成本，也提高了交互效率”改写得更像论文语言。",
    },
    {
        "prompt_id": "P08_classical_compressed",
        "text": "请将“这项研究很有意义，因为它不仅节省成本，也提高了交互效率”改写为较近论文之语。",
    },
    {
        "prompt_id": "P09_original",
        "text": "请给我一个3天完成课程论文初稿的时间安排表，按每天上午、下午、晚上分段列出任务。",
    },
    {
        "prompt_id": "P09_free_compressed",
        "text": "给我一个3天完成课程论文初稿的时间安排表，按每天上午、下午、晚上分段列任务。",
    },
    {
        "prompt_id": "P09_classical_compressed",
        "text": "请拟三日内完成课程论文初稿之日程表，依每日上午、下午、晚上分列任务。",
    },
    {
        "prompt_id": "P10_original",
        "text": "请用通俗中文解释“高信息密度表达”和“表达清晰”之间可能存在什么张力，并举一个简单例子。",
    },
    {
        "prompt_id": "P10_free_compressed",
        "text": "用通俗中文解释“高信息密度表达”和“表达清晰”之间可能存在的张力，并举一个简单例子。",
    },
    {
        "prompt_id": "P10_classical_compressed",
        "text": "请以通俗中文解释“高信息密度表达”与“表达清晰”之间可能之张力，并举一简例。",
    },
    {
        "prompt_id": "P11_original",
        "text": "请把“我们的方法可能有效，但还需要更多实验验证”改写成更谨慎、更像论文讨论部分的表达。",
    },
    {
        "prompt_id": "P11_free_compressed",
        "text": "把“我们的方法可能有效，但还需要更多实验验证”改写得更谨慎，更像论文讨论部分。",
    },
    {
        "prompt_id": "P11_classical_compressed",
        "text": "请将“我们的方法可能有效，但还需要更多实验验证”改写为更审慎、较近论文讨论之语。",
    },
    {
        "prompt_id": "P12_original",
        "text": "请为一个关于“文言文压缩与大模型交互成本”的研究设计5个可能的实验指标，并各用一句话说明其意义。",
    },
    {
        "prompt_id": "P12_free_compressed",
        "text": "为“文言文压缩与大模型交互成本”设计5个实验指标，并各用一句话说明意义。",
    },
    {
        "prompt_id": "P12_classical_compressed",
        "text": "请为“文言文压缩与大模型交互成本”设计实验指标五项，并各以一句略述其义。",
    },
    {
        "prompt_id": "P13_original",
        "text": "请把下面这句话翻译成自然英文，可用于论文摘要中的一句话：本文比较原始prompt、自由压缩prompt和文言式压缩prompt在token成本与任务表现上的差异。",
    },
    {
        "prompt_id": "P13_free_compressed",
        "text": "把这句话译成自然英文，可用于论文摘要：本文比较原始prompt、自由压缩prompt和文言式压缩prompt在token成本与任务表现上的差异。",
    },
    {
        "prompt_id": "P13_classical_compressed",
        "text": "请将此句译为自然英文，可用于论文摘要：本文比较原始prompt、自由压缩prompt与文言式压缩prompt在token成本及任务表现上的差异。",
    },
    {
        "prompt_id": "P14_original",
        "text": "请把下面这句话翻译成自然英文，可用于写给国际会议组织者的邮件：由于签证办理时间较长，我想申请延长注册截止日期。",
    },
    {
        "prompt_id": "P14_free_compressed",
        "text": "把这句话译成自然英文，可用于写给国际会议组织者的邮件：由于签证办理时间较长，我想申请延长注册截止日期。",
    },
    {
        "prompt_id": "P14_classical_compressed",
        "text": "请将此句译为自然英文，可用于致国际会议主办方之邮件：因签证办理时日较长，我欲申请延长注册截止日期。",
    },
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
    return os.environ.get("OPENAI_API_KEY")


def build_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key:
        print("错误：未检测到环境变量 OPENAI_API_KEY。")
        print("请先在 PowerShell 中执行：")
        print('  setx OPENAI_API_KEY "你的_API_Key"')
        print("然后关闭并重新打开 PowerShell。")
        raise SystemExit(1)
    return OpenAI(api_key=api_key)


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


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def usage_to_dict(usage: Any) -> Dict[str, Any]:
    input_tokens = safe_getattr(usage, "input_tokens", None)
    output_tokens = safe_getattr(usage, "output_tokens", None)
    total_tokens = safe_getattr(usage, "total_tokens", None)

    output_details = safe_getattr(usage, "output_tokens_details", None)
    reasoning_tokens = safe_getattr(output_details, "reasoning_tokens", None)

    if reasoning_tokens is None and isinstance(output_details, dict):
        reasoning_tokens = output_details.get("reasoning_tokens")

    return {
        "response_prompt_token_count": input_tokens,
        "response_output_token_count": output_tokens,
        "response_total_token_count": total_tokens,
        "response_reasoning_token_count": reasoning_tokens,
    }


def extract_text(response: Any) -> str:
    output_text = safe_getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: List[str] = []
    output_items = safe_getattr(response, "output", []) or []
    for item in output_items:
        content_list = safe_getattr(item, "content", []) or []
        for content in content_list:
            text_val = safe_getattr(content, "text", None)
            if isinstance(text_val, str) and text_val.strip():
                texts.append(text_val.strip())
    return "\n".join(texts).strip()


def run_single_request(
    client: OpenAI,
    model_name: str,
    prompt_id: str,
    prompt_text: str,
    successful_run_index: int,
    global_attempt_index: int,
) -> Dict[str, Any]:
    last_error: Optional[str] = None

    for retry_index in range(1, MAX_RETRIES_PER_ATTEMPT + 1):
        started_at = datetime.now().isoformat(timespec="seconds")
        try:
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "input": prompt_text,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
            if REASONING_EFFORT:
                kwargs["reasoning"] = {"effort": REASONING_EFFORT}

            response = client.responses.create(**kwargs)

            usage = usage_to_dict(safe_getattr(response, "usage", None))
            output_text = extract_text(response)

            return {
                "prompt_id": prompt_id,
                "run_index": successful_run_index,
                "global_attempt_index": global_attempt_index,
                "retry_index": retry_index,
                "status": "ok",
                "started_at": started_at,
                "model_name": model_name,
                "prompt_text": prompt_text,
                **usage,
                "output_text": output_text,
                "error": "",
                "response_id": safe_getattr(response, "id", None),
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
                    "response_prompt_token_count": None,
                    "response_output_token_count": None,
                    "response_total_token_count": None,
                    "response_reasoning_token_count": None,
                    "output_text": "",
                    "error": last_error,
                    "response_id": None,
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
                "mean_response_prompt_token_count": mean_of(
                    "response_prompt_token_count", items
                ),
                "mean_response_output_token_count": mean_of(
                    "response_output_token_count", items
                ),
                "mean_response_total_token_count": mean_of(
                    "response_total_token_count", items
                ),
                "mean_response_reasoning_token_count": mean_of(
                    "response_reasoning_token_count", items
                ),
                "error_rows": sum(
                    1 for item in rows
                    if item.get("prompt_id") == prompt_id and item.get("status") != "ok"
                ),
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
        "response_prompt_token_count: {0}".format(row.get("response_prompt_token_count", "")),
        "response_output_token_count: {0}".format(row.get("response_output_token_count", "")),
        "response_total_token_count: {0}".format(row.get("response_total_token_count", "")),
        "response_reasoning_token_count: {0}".format(row.get("response_reasoning_token_count", "")),
        "response_id: {0}".format(row.get("response_id", "")),
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
    all_rows: List[Dict[str, Any]] = []

    print("开始运行：{0} 个 prompt；每个目标成功 {1} 次。".format(len(prompts), SUCCESSFUL_RUNS_PER_PROMPT))
    print("模型：{0}".format(MODEL_NAME))
    print("提示：本脚本使用的是单轮 responses.create，不会自动跨轮累加上下文。")
    print("提示：当前 REASONING_EFFORT = {0}".format(REASONING_EFFORT))
    print("提示：当前 temperature = {0}, top_p = {1}, max_output_tokens = {2}".format(
        TEMPERATURE, TOP_P, MAX_OUTPUT_TOKENS
    ))
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
    print("2) 只要你不把 previous_response_id 传进来，上下文就不会自动带入下一轮。")
    print("3) 正式统计时优先用 response_prompt_token_count / response_output_token_count / response_total_token_count。")
    print("4) 如果某个 prompt 最终没拿满 3 次成功，请先看 run_level_results.csv 里的 error 列。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
