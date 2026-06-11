"""
POST /parse — OCR 解析
GET  /health — 健康检查
"""

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile
from mineru.cli.common import aio_do_parse

from src.config import settings
from src.schemas import HealthResponse, ParseResponse

router = APIRouter()

# 合法的 task 值
VALID_TASKS = {"default", "double_page"}


@router.post("/parse", response_model=ParseResponse)
async def parse(
    file: UploadFile = Form(...),
    task: str = Form(default="default"),
):
    """接收文档文件，返回全文 + 分页内容。"""

    # ---- 1. 参数校验 ----
    if task not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"task 取值必须为 {sorted(VALID_TASKS)}，当前值: {task!r}",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")

    file_bytes = await file.read()
    if len(file_bytes) > settings.max_file_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超过限制 ({settings.max_file_mb} MB)",
        )

    # ---- 2. 准备 MinerU 参数 ----
    task_id = uuid.uuid4().hex[:12]
    file_stem = Path(file.filename).stem
    output_dir = Path(settings.output_dir) / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 双页排版：如果 MinerU 内部不支持，可在此用 PyMuPDF 预切页后重新喂入
    # 当前先统一透传，后续根据实际效果调整。
    parse_method = settings.parse_method
    if task == "double_page":
        # TODO: 双页模式可能需要 PDF 页面竖切预处理，或调整 MinerU 的 layout 参数
        pass

    # ---- 3. 调用 MinerU 引擎 ----
    try:
        await aio_do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[file_stem],
            pdf_bytes_list=[file_bytes],
            p_lang_list=[settings.default_lang],
            backend=settings.default_backend,
            parse_method=parse_method,
            formula_enable=True,
            table_enable=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"MinerU 解析失败: {exc}",
        )

    # ---- 4. 收集结果 ----
    result_dir = output_dir / file_stem
    md_path = result_dir / f"{file_stem}.md"
    cl_path = result_dir / f"{file_stem}_content_list.json"

    if not md_path.exists():
        # 尝试不带 stem 的目录结构（MinerU 版本差异）
        alt_md = output_dir / f"{file_stem}.md"
        alt_cl = output_dir / f"{file_stem}_content_list.json"
        if alt_md.exists():
            md_path = alt_md
            cl_path = alt_cl
        else:
            raise HTTPException(
                status_code=500,
                detail="MinerU 解析完成但未生成预期输出文件",
            )

    # 全文
    content = md_path.read_text(encoding="utf-8")

    # 分页内容
    pages_content: list[str] = []
    if cl_path.exists():
        try:
            with open(cl_path, encoding="utf-8") as f:
                blocks = json.load(f)
            # 按 page_idx 聚合
            page_map: dict[int, list[str]] = {}
            for block in blocks:
                page_idx = block.get("page_idx", 0)
                text = block.get("text") or block.get("md") or ""
                if text.strip():
                    page_map.setdefault(page_idx, []).append(text)
            pages_content = [
                "\n".join(page_map[i])
                for i in sorted(page_map)
            ]
        except (json.JSONDecodeError, KeyError):
            pages_content = []

    # ---- 5. 清理落盘 ----
    if settings.cleanup_after_parse:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

    return ParseResponse(content=content, pages_content=pages_content)


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        backend=settings.default_backend,
        max_file_mb=settings.max_file_mb,
    )
