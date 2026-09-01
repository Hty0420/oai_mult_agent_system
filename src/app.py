import asyncio
import threading
import os
import shutil
import re
import gradio as gr
import queue
import time
import uuid

from oai_client import OAI_ChatBot
from db_utils.db_utils import DatabaseManager
from db_utils.trajectory import current_policy_id, personal_workspace_id
from db_utils.workspace_memory import WORKSPACE_MEMORY_TYPE_LABELS
from mesh_visualization import (
    build_mesh_preview,
    collect_structured_output_files,
    find_mesh_manifest,
    MeshRefreshTracker,
)
from agents.tool_progress import append_progress_text, progress_line_key
from system_capabilities import render_system_capabilities

# ================== 全局变量 ==================
USE_FILE_DIR = "user_upload_file"
if not os.path.exists(USE_FILE_DIR):
    os.makedirs(USE_FILE_DIR)

DEFAULT_IMAGE_PATH = "../assets/default_placeholder.png"

global_chatbot = None
loop = None
is_initialized = False
current_user = None
current_session_id = None
current_workspace_id = None
capability_overview_html, capability_agents_html, capability_tools_html = (
    render_system_capabilities()
)

chat_history = []
chart_data = []
current_charts = []
available_download_files = []  # 存储所有可用的下载文件
background_futures = {}
background_futures_lock = threading.Lock()
background_submission_lock = threading.Lock()
background_execution_lock = None
MAX_PERSISTED_PROGRESS_CHARS = 60000
mesh_refresh_tracker = MeshRefreshTracker()

# ================== 登录逻辑 ==================
def login(username, password):
    if not username or not password:
        return "❌ 请输入用户名和密码"

    db_manager = DatabaseManager()
    user_id = db_manager.get_or_create_user_id(username)
    db_manager.close()

    if user_id:
        # ✅ 登录成功后，返回一个 HTML 跳转
        html = f"""
        <div>
            ✅ 登录成功，欢迎 <b>{username}</b> ！正在跳转到聊天页面...
            <meta http-equiv="refresh" content="1; url=/set_session?username={username}">
        </div>
        """
        return html
    else:
        return "❌ 登录失败，用户不存在"


# ================== 工具列表 ==================
def update_tools_display():
    if global_chatbot is None:
        return render_system_capabilities(initialized=False)
    return render_system_capabilities(
        global_chatbot.available_tools,
        initialized=is_initialized,
        multi_agent=global_chatbot.use_multi_agent,
    )


# ================== 文件保存 ==================
def save_single_file(file):
    if file is None:
        return None
    filename = os.path.basename(file.name)
    save_path = os.path.join(USE_FILE_DIR, filename)
    try:
        if hasattr(file, "read"):
            content = file.read()
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(save_path, mode) as f:
                f.write(content)
            return save_path
        else:
            shutil.copy(file.name, save_path)
            return save_path
    except Exception as e:
        print(f"保存文件失败: {str(e)}")
        return None


def _log_user_files(
    file_paths,
    file_type,
    source_tool=None,
    user_id=None,
    session_id=None,
):
    """将文件信息记录到 user_files 表"""
    try:
        if user_id is None or session_id is None:
            if not is_initialized or global_chatbot is None:
                return
            user_id = getattr(global_chatbot, 'user_id', None)
            session_id = getattr(global_chatbot, 'session_id', None)
        if not user_id or not session_id:
            return
        db = DatabaseManager()
        for path in file_paths:
            if path and os.path.exists(path):
                file_name = os.path.basename(path)
                db.save_user_file(
                    user_id=user_id,
                    session_id=session_id,
                    file_name=file_name,
                    file_path=path,
                    file_type=file_type,
                    source_tool=source_tool
                )
        db.close()
    except Exception as e:
        print(f"记录用户文件失败: {e}")


def save_uploaded_files(files, user_id=None, session_id=None):
    if not files:
        return []
    saved_paths = []
    for file in files:
        path = save_single_file(file)
        if path:
            saved_paths.append(path)
    # 记录上传文件
    _log_user_files(
        saved_paths,
        file_type='upload',
        user_id=user_id,
        session_id=session_id,
    )
    return saved_paths


# ================== 图表解析 ==================
def _chart_identity(chart):
    """Return a stable identity for a chart without changing its display value."""
    chart_type = str(chart.get("type") or "")
    if chart_type == "image":
        # The payload uniquely identifies an embedded image even if it appears
        # once in streamed progress and again in the final response.
        data = str(chart.get("data") or "")
        return ("image", data.partition(",")[2] or data)
    if chart_type == "file":
        path = str(chart.get("path") or "").strip()
        # realpath + normcase also merges relative/absolute, slash and case
        # variants that point at the same chart on Windows.
        return (
            "file",
            os.path.normcase(os.path.realpath(os.path.abspath(path))),
        )
    return (chart_type, repr(chart))


def _deduplicate_charts(charts):
    unique_charts = []
    seen = set()
    for chart in charts or []:
        if not isinstance(chart, dict):
            continue
        identity = _chart_identity(chart)
        if identity in seen:
            continue
        seen.add(identity)
        unique_charts.append(chart)
    return unique_charts


def parse_chart_data(response):
    charts = []
    try:
        if "data:image" in response:
            pattern = r"data:image/(png|jpg|jpeg|gif);base64,([A-Za-z0-9+/=]+)"
            matches = re.findall(pattern, response)
            for format_type, data in matches:
                charts.append({
                    "type": "image",
                    "format": format_type,
                    "data": f"data:image/{format_type};base64,{data}"
                })
        if "file_path" in response.lower():
            pattern = r'file_path["\s]*:["\s]*([^"\n]+)'
            matches = re.findall(pattern, response)
            for path in matches:
                # Only include image files — skip txt/csv to avoid broken gallery items
                if os.path.exists(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    charts.append({"type": "file", "path": path})
    except Exception as e:
        print(f"解析图表数据时出错: {str(e)}")
    return _deduplicate_charts(charts)


def load_default_image():
    # 始终返回不可见状态，不加载任何默认图片
    return gr.update(visible=False), []

def load_charts(charts):
    if not charts:
        return gr.update(visible=False), []
    images = []
    # Keep this second boundary defensive: callers other than
    # parse_chart_data may also provide repeated chart entries.
    for chart in _deduplicate_charts(charts):
        if chart["type"] == "image":
            images.append(chart["data"])
        elif chart["type"] == "file" and os.path.exists(chart["path"]):
            images.append(chart["path"])
    count = len(images)
    if count == 1:
        columns, rows, height = 1, 1, 440
    elif count == 2:
        columns, rows, height = 1, 2, 620
    else:
        columns, rows, height = 1, min(count, 3), 680
    return gr.update(visible=True, columns=columns, rows=rows, height=height), images


# ================== 文件解析 ==================
def parse_output_files(response):
    output_files = collect_structured_output_files(response)
    try:
        # 1. 处理分割文件相关的响应
        if "umamba_segment_mri" in response or "lkm_unet_segment_mri" in response or "lightm_unet_segment_mri" in response or "分割文件已保存成功" in response:
            pattern = r'"files"\s*:\s*\[([^\]]+)\]'
            matches = re.findall(pattern, response)
            if matches:
                paths = [p.strip().strip('"') for p in matches[0].split(",")]
                for path in paths:
                    # 统一路径格式，将所有反斜杠替换为斜杠
                    path = path.replace('\\', '/').strip()
                    if os.path.exists(path):
                        output_files.append(path)
        # 2. 增强的正则表达式模式，确保捕获 CSV 文件
        # 原始模式 - 匹配带文件/路径前缀的文件名
        pattern1 = r"(?:文件|路径)[:：]\s*([^,\n]+\.(?:zip|nii\.gz|csv|txt|png))"
        matches1 = re.findall(pattern1, response)
        
        # 额外模式 - 专门捕获 CSV 和 TXT 文件
        pattern_csv_txt = r"([^,\n]+\.(?:csv|txt))"
        matches_csv_txt = re.findall(pattern_csv_txt, response)
        
        # 额外模式 - 匹配可能包含目录路径的CSV和TXT文件
        pattern_path = r"(?:\/|\\)([^,\n]+\.(?:csv|txt))"
        matches_path = re.findall(pattern_path, response)
        
        # 3. 合并所有匹配结果
        all_matches = matches1 + matches_csv_txt + matches_path
        
        # 4. 处理所有匹配的文件路径
        for path in all_matches:
            # 确保路径不包含多余的空格
            path = path.strip()
            # 统一路径格式，将所有反斜杠替换为斜杠
            path = path.replace('\\', '/').strip()
            # 检查文件是否存在且尚未添加
            if os.path.exists(path):
                # 确保路径不重复添加（使用标准化后的路径比较）
                normalized_path = path.replace('\\', '/')
                if not any(f.replace('\\', '/') == normalized_path for f in output_files):
                    output_files.append(path)
                
        # 5. 特殊处理CSV和TXT文件 - 检查常见输出目录
        file_keywords = ["features", "summary", "result", "output", "radiomics", "results", "log", "report", "data"]
        common_dirs = ["output", "results", "result", "logs", "reports", "data"]
        
        for dir_path in common_dirs:
            # 统一目录路径格式
            dir_path = dir_path.replace('\\', '/')
            if os.path.isdir(dir_path):
                for filename in os.listdir(dir_path):
                    if (filename.lower().endswith(".csv") or filename.lower().endswith(".txt")) and \
                    any(keyword in filename.lower() for keyword in file_keywords):
                        # 构建完整路径并统一格式
                        full_path = os.path.join(dir_path, filename).replace('\\', '/')
                        if os.path.exists(full_path):
                            # 标准化路径用于比较，避免重复添加
                            normalized_full_path = full_path.replace('\\', '/')
                            if not any(f.replace('\\', '/') == normalized_full_path for f in output_files):
                                # 只有当响应中提及了这个文件或目录时才添加
                                if filename in response or dir_path.replace('/', '') in response:
                                    output_files.append(full_path)

        # 6. 增强功能：直接搜索响应中提到的CSV和TXT文件名
        # 提取所有可能的文件名
        all_filenames = re.findall(r'"([^"]+\.(?:csv|txt))"', response)
        all_filenames += re.findall(r'\b([^\s"\',]+\.(?:csv|txt))\b', response)
        
        # 遍历所有可能的文件名目录
        search_dirs = ["output", "results", "user_upload_file","result"]
        
        for filename in all_filenames:
            for search_dir in search_dirs:
                # 统一目录路径格式
                search_dir = search_dir.replace('\\', '/')
                # 构建完整路径
                full_path = os.path.join(search_dir, filename).replace('\\', '/')
                # 检查文件是否存在且尚未添加
                if os.path.exists(full_path):
                    # 标准化路径用于比较，避免重复添加
                    normalized_full_path = full_path.replace('\\', '/')
                    if not any(f.replace('\\', '/') == normalized_full_path for f in output_files):
                        output_files.append(full_path)
        
        # 7. 最终去重和标准化路径
        unique_files = []
        added_paths = set()
        
        for file_path in output_files:
            # 标准化路径
            normalized_path = file_path.replace('\\', '/')
            # 确保不重复添加
            if normalized_path not in added_paths:
                added_paths.add(normalized_path)
                unique_files.append(file_path)
        
        output_files = unique_files
    except Exception as e:
        print(f"解析输出文件时出错: {str(e)}")
    
    return output_files

# ================== 聊天机器人 ==================
def _chatbot_history_pairs(messages):
    pairs = []
    pending_user = None
    for message in messages or []:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            if pending_user is not None:
                pairs.append((pending_user, ""))
            pending_user = content
        elif role == "assistant":
            pairs.append((pending_user, content))
            pending_user = None
    if pending_user is not None:
        pairs.append((pending_user, ""))
    return pairs


def _workspace_choices(db, user_id):
    return [
        (row["display_name"], row["workspace_id"])
        for row in db.list_workspaces(user_id)
    ]


def _next_workspace_display_name(workspaces):
    """Generate a concise unique name without adding another manual form."""
    existing = {
        str(row.get("display_name") or "").strip()
        for row in workspaces or []
    }
    index = 1
    while f"研究工作区 {index}" in existing:
        index += 1
    return f"研究工作区 {index}"


def _format_session_label(session):
    updated_at = session.get("updated_at") or session.get("created_at")
    if hasattr(updated_at, "strftime"):
        time_label = updated_at.strftime("%m-%d %H:%M")
    else:
        time_label = str(updated_at or "")[:16].replace("T", " ") or "刚刚"
    return f"{session.get('title') or '新会话'}  ·  {time_label}"


def _session_selector_state(selected_session_id=None):
    if not is_initialized or global_chatbot is None:
        return gr.update(choices=[], value=None)
    db = DatabaseManager()
    try:
        sessions = db.list_workspace_sessions(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            limit=100,
        )
    finally:
        db.close()
    choices = [
        (_format_session_label(session), session["session_id"])
        for session in sessions
    ]
    selected = selected_session_id or global_chatbot.session_id
    if selected not in {value for _, value in choices}:
        selected = choices[0][1] if choices else None
    return gr.update(choices=choices, value=selected)


def workspace_dashboard_markdown():
    if not is_initialized or global_chatbot is None:
        return "### 自优化轨迹\n\n工作台加载后将自动汇总当前工作区的运行轨迹。"
    db = DatabaseManager()
    try:
        summary = db.get_workspace_optimization_summary(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            global_chatbot.agent_version,
        )
        workspaces = {
            row["workspace_id"]: row["display_name"]
            for row in db.list_workspaces(global_chatbot.user_id)
        }
    finally:
        db.close()
    workspace_name = workspaces.get(
        global_chatbot.workspace_id, global_chatbot.workspace_id
    )
    counts = summary["partitions"]
    train = counts.get("train", {})
    validation = counts.get("validation", {})
    test = counts.get("test", {})
    latest = summary.get("latest_task") or {}
    latest_text = "暂无任务轨迹"
    if latest:
        trace_id = str(latest.get("trace_id") or "")
        latest_text = (
            f"`{trace_id[:8]}…` · `{latest.get('data_partition')}` · "
            f"`{latest.get('status')}`"
        )
    return (
        "### 自优化轨迹\n\n"
        f"**{workspace_name}**\n\n"
        f"`{global_chatbot.agent_version}` · `{current_policy_id()}`\n\n"
        "| Train | Validation | Test |\n"
        "|---:|---:|---:|\n"
        f"| {train.get('task_count', 0)} | "
        f"{validation.get('task_count', 0)} | {test.get('task_count', 0)} |\n\n"
        "> Train 生成候选 · Validation 验证 · Test 冻结评估\n\n"
        f"已完成：`{train.get('completed_count', 0) + validation.get('completed_count', 0) + test.get('completed_count', 0)}`"
        f"　反馈轨迹：`{summary['human_feedback_count']}`\n\n"
        f"最近任务：{latest_text}"
    )


def _memory_panel_state():
    if not is_initialized or global_chatbot is None:
        return gr.update(choices=[], value=None), (
            "尚无可显示的工作区记忆。初始化后可为当前工作区添加长期记忆。"
        )
    db = DatabaseManager()
    try:
        records = db.list_workspace_memories(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            active_only=True,
            limit=100,
        )
    finally:
        db.close()
    choices = []
    lines = []
    for record in records:
        label = WORKSPACE_MEMORY_TYPE_LABELS.get(
            record["memory_type"], record["memory_type"]
        )
        content = str(record["content"])
        short_content = content.replace("\n", " ")[:54]
        choices.append((f"[{label}] {short_content}", str(record["id"])))
        safe_content = content.replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f"- **{label}**：{safe_content}")
    preview = (
        "当前工作区还没有长期记忆。可添加项目事实、工作偏好或长期指令。"
        if not lines
        else "#### 当前有效记忆\n\n" + "\n".join(lines)
    )
    return gr.update(choices=choices, value=None), preview


def load_workspace_memory_for_edit(memory_id):
    if not memory_id or not is_initialized or global_chatbot is None:
        return "project_fact", "", "请选择一条记忆。"
    db = DatabaseManager()
    try:
        record = db.get_workspace_memory(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            int(memory_id),
        )
    finally:
        db.close()
    if not record or not record.get("active"):
        return "project_fact", "", "该记忆不存在或已经遗忘。"
    return record["memory_type"], record["content"], "已载入，可修改或遗忘。"


def add_workspace_memory(memory_type, content):
    if not is_initialized or global_chatbot is None:
        selector, preview = _memory_panel_state()
        return selector, preview, "请先初始化系统。", content
    db = DatabaseManager()
    try:
        memory_id = db.add_workspace_memory(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            memory_type,
            content,
            source="human",
        )
    except Exception as exc:
        return gr.update(), _memory_panel_state()[1], f"添加失败：{exc}", content
    finally:
        db.close()
    selector, preview = _memory_panel_state()
    return selector, preview, f"已添加记忆 #{memory_id}。", ""


def update_workspace_memory(memory_id, memory_type, content):
    if not memory_id or not is_initialized or global_chatbot is None:
        return gr.update(), _memory_panel_state()[1], "请先选择一条记忆。"
    db = DatabaseManager()
    try:
        updated = db.update_workspace_memory(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            int(memory_id),
            memory_type,
            content,
        )
    except Exception as exc:
        return gr.update(), _memory_panel_state()[1], f"更新失败：{exc}"
    finally:
        db.close()
    selector, preview = _memory_panel_state()
    status = "记忆已更新。" if updated else "记忆不存在或已经遗忘。"
    return selector, preview, status


def forget_workspace_memory(memory_id):
    if not memory_id or not is_initialized or global_chatbot is None:
        selector, preview = _memory_panel_state()
        return selector, preview, "请先选择一条记忆。", "project_fact", ""
    db = DatabaseManager()
    try:
        forgotten = db.forget_workspace_memory(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            int(memory_id),
        )
    except Exception as exc:
        selector, preview = _memory_panel_state()
        return selector, preview, f"遗忘失败：{exc}", "project_fact", ""
    finally:
        db.close()
    selector, preview = _memory_panel_state()
    status = "该记忆已遗忘。" if forgotten else "记忆不存在或已经遗忘。"
    return selector, preview, status, "project_fact", ""


def _ensure_execution_loop():
    """Return the process-level loop that owns all durable background jobs."""
    global loop
    if loop is not None and loop.is_running() and not loop.is_closed():
        return loop
    execution_loop = asyncio.new_event_loop()
    loop = execution_loop

    def run_loop():
        asyncio.set_event_loop(execution_loop)
        execution_loop.run_forever()

    threading.Thread(target=run_loop, daemon=True).start()
    return loop


def _get_active_background_task(user_id):
    if not user_id:
        return None
    db = DatabaseManager()
    try:
        return db.get_active_background_task(user_id)
    finally:
        db.close()


def _composer_updates(active_run=None, clear=False, enabled=True):
    """Build matching textbox/upload updates from the persisted job state."""
    if active_run:
        run_status = active_run.get("status")
        label = "正在执行" if run_status == "running" else "正在排队"
        session_id = str(active_run.get("session_id") or "")
        suffix = f"（会话 {session_id[:8]}…）" if session_id else ""
        text_kwargs = {
            "interactive": False,
            "placeholder": f"当前任务{label}{suffix}，完成后可提交新任务",
        }
        if clear:
            text_kwargs["value"] = ""
        file_update = (
            gr.update(value=None, interactive=False)
            if clear
            else gr.update(interactive=False)
        )
        return gr.update(**text_kwargs), file_update
    text_kwargs = {
        "interactive": bool(enabled),
        "placeholder": "输入要完成的任务，按 Enter 发送给 Agent…",
    }
    if clear:
        text_kwargs["value"] = ""
    return gr.update(**text_kwargs), gr.update(interactive=bool(enabled))


def _active_task_status(active_run, current_session_id=None):
    run_status = active_run.get("status")
    label = "后台执行中" if run_status == "running" else "等待后台执行"
    active_session_id = str(active_run.get("session_id") or "")
    location = "当前会话"
    if current_session_id and active_session_id != current_session_id:
        location = f"会话 {active_session_id[:8]}…"
    progress_lines = [
        line.strip()
        for line in str(active_run.get("progress_text") or "").splitlines()
        if progress_line_key(line)
    ]
    progress_suffix = f" · {progress_lines[-1]}" if progress_lines else ""
    return f"任务{label} · {location}{progress_suffix} · 完成前不可提交新任务"


def initialize_chatbot():
    global global_chatbot, loop, is_initialized, current_user, current_session_id, current_workspace_id
    if is_initialized:
        db = DatabaseManager()
        try:
            choices = _workspace_choices(db, global_chatbot.user_id)
            active_run = db.get_active_background_task(global_chatbot.user_id)
        finally:
            db.close()
        pairs, runtime_status, _, _, _ = _session_runtime_view(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            global_chatbot.session_id,
        )
        input_update, file_update = _composer_updates(active_run)
        return (
            (
                _active_task_status(active_run, global_chatbot.session_id)
                if active_run
                else runtime_status or "聊天机器人已经初始化"
            ),
            gr.update(choices=choices, value=global_chatbot.workspace_id),
            workspace_dashboard_markdown(),
            pairs,
            _session_selector_state(global_chatbot.session_id),
            input_update,
            file_update,
        )
    if not current_user:
        input_update, file_update = _composer_updates(enabled=False)
        return (
            "请先登录",
            gr.update(),
            workspace_dashboard_markdown(),
            [],
            gr.update(choices=[], value=None),
            input_update,
            file_update,
        )
    db_manager = DatabaseManager()
    user_id = db_manager.get_or_create_user_id(current_user)
    current_workspace_id = db_manager.ensure_personal_workspace(user_id)
    current_session_id = db_manager.get_user_default_session_id(
        user_id, current_workspace_id
    )
    choices = _workspace_choices(db_manager, user_id)
    active_run = db_manager.get_active_background_task(user_id)
    db_manager.close()
    _ensure_execution_loop()
    global_chatbot = OAI_ChatBot(
        username=current_user,
        use_multi_agent=True,
        workspace_id=current_workspace_id,
    )
    global_chatbot.session_id = current_session_id
    global_chatbot._load_history()

    chatbot_instance = global_chatbot

    async def connect():
        await chatbot_instance.connect_to_servers()
        return f"系统已初始化 · 用户：{current_user} · 工作区：个人工作区"

    future = asyncio.run_coroutine_threadsafe(connect(), loop)
    is_initialized = True
    try:
        result = future.result(timeout=1.5)
    except TimeoutError:
        result = f"工作台已就绪 · 用户：{current_user} · 工具正在后台连接"

        def report_background_connection(done_future):
            try:
                done_future.result()
            except Exception as exc:
                print(f"后台工具连接失败: {exc or type(exc).__name__}")

        future.add_done_callback(report_background_connection)
    except Exception as e:
        detail = str(e) or type(e).__name__
        result = f"工作台已载入 · 部分工具连接失败：{detail}"
    pairs, runtime_status, _, _, _ = _session_runtime_view(
        user_id, current_workspace_id, current_session_id
    )
    input_update, file_update = _composer_updates(active_run)
    if active_run:
        result = _active_task_status(active_run, current_session_id)
    elif runtime_status:
        result = runtime_status
    return (
        result,
        gr.update(choices=choices, value=current_workspace_id),
        workspace_dashboard_markdown(),
        pairs,
        _session_selector_state(current_session_id),
        input_update,
        file_update,
    )


def switch_workspace(workspace_id):
    global current_workspace_id, current_session_id, available_download_files
    if not is_initialized or global_chatbot is None:
        input_update, file_update = _composer_updates(enabled=False)
        return (
            [],
            "请先初始化系统",
            workspace_dashboard_markdown(),
            gr.update(choices=[]),
            gr.update(choices=[], value=None),
            input_update,
            file_update,
            gr.update(),
            gr.update(),
        )
    if not workspace_id:
        active_run = _get_active_background_task(global_chatbot.user_id)
        input_update, file_update = _composer_updates(active_run)
        return (
            _chatbot_history_pairs(global_chatbot.chat_history),
            "请选择一个工作区",
            workspace_dashboard_markdown(),
            gr.update(),
            _session_selector_state(global_chatbot.session_id),
            input_update,
            file_update,
            gr.update(),
            gr.update(),
        )
    db = DatabaseManager()
    try:
        if not db.workspace_belongs_to_user(global_chatbot.user_id, workspace_id):
            raise ValueError("工作区不存在或不属于当前用户")
        current_workspace_id = workspace_id
        current_session_id = db.get_user_default_session_id(
            global_chatbot.user_id, workspace_id
        )
        active_run = db.get_active_background_task(global_chatbot.user_id)
    finally:
        db.close()
    global_chatbot.workspace_id = workspace_id
    global_chatbot.session_id = current_session_id
    global_chatbot.last_run_metadata = {}
    global_chatbot._load_history()
    available_download_files = []
    pairs, runtime_status, gallery_update, chart_images, downloads = _session_runtime_view(
        global_chatbot.user_id, workspace_id, current_session_id
    )
    input_update, file_update = _composer_updates(active_run)
    return (
        pairs,
        (
            _active_task_status(active_run, current_session_id)
            if active_run
            else runtime_status or f"已切换工作区 · 会话 {current_session_id[:8]}…"
        ),
        workspace_dashboard_markdown(),
        downloads,
        _session_selector_state(current_session_id),
        input_update,
        file_update,
        gallery_update,
        chart_images,
    )


def _workspace_action_noop():
    return tuple(gr.update() for _ in range(10))


def create_new_workspace(display_name=None):
    """Create, activate and initialize an isolated workspace in one action."""
    global current_workspace_id, current_session_id, available_download_files
    if display_name is None:
        return _workspace_action_noop()
    display_name = str(display_name).strip()
    if not display_name:
        return _workspace_action_noop()
    if not is_initialized or global_chatbot is None:
        input_update, file_update = _composer_updates(enabled=False)
        return (
            gr.update(), [], "请先初始化系统", workspace_dashboard_markdown(),
            gr.update(choices=[], value=None, visible=False),
            gr.update(choices=[], value=None), input_update, file_update,
            gr.update(visible=False), [],
        )

    db = DatabaseManager()
    try:
        workspace_id = db.create_workspace(
            global_chatbot.user_id, display_name
        )
        session_id = db.get_user_default_session_id(
            global_chatbot.user_id, workspace_id
        )
        choices = _workspace_choices(db, global_chatbot.user_id)
        active_run = db.get_active_background_task(global_chatbot.user_id)
    finally:
        db.close()

    current_workspace_id = workspace_id
    current_session_id = session_id
    global_chatbot.workspace_id = workspace_id
    global_chatbot.session_id = session_id
    global_chatbot.chat_history = []
    global_chatbot.last_run_metadata = {}
    available_download_files = []

    pairs, runtime_status, gallery_update, chart_images, downloads = (
        _session_runtime_view(
            global_chatbot.user_id, workspace_id, session_id
        )
    )
    input_update, file_update = _composer_updates(active_run)
    status = (
        _active_task_status(active_run, session_id)
        if active_run
        else runtime_status or f"已创建并进入 {display_name}"
    )
    return (
        gr.update(choices=choices, value=workspace_id),
        pairs,
        status,
        workspace_dashboard_markdown(),
        downloads,
        _session_selector_state(session_id),
        input_update,
        file_update,
        gallery_update,
        chart_images,
    )


def rename_current_workspace(workspace_id, display_name=None):
    """Rename the selected workspace without disturbing its active session."""
    if not workspace_id or display_name is None:
        return _workspace_action_noop()
    display_name = str(display_name).strip()
    if not display_name:
        return _workspace_action_noop()
    if not is_initialized or global_chatbot is None:
        input_update, file_update = _composer_updates(enabled=False)
        return (
            gr.update(), [], "请先初始化系统", workspace_dashboard_markdown(),
            gr.update(), gr.update(), input_update, file_update,
            gr.update(), gr.update(),
        )

    db = DatabaseManager()
    try:
        renamed = db.rename_workspace(
            global_chatbot.user_id, str(workspace_id), display_name
        )
        choices = _workspace_choices(db, global_chatbot.user_id)
    except Exception as exc:
        return (
            gr.update(), gr.update(),
            f"修改工作区名称失败：{str(exc) or type(exc).__name__}",
            workspace_dashboard_markdown(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
        )
    finally:
        db.close()

    return (
        gr.update(choices=choices, value=workspace_id),
        gr.update(),
        (
            f"工作区已重命名为 {display_name}"
            if renamed
            else "工作区已不存在，名称未修改"
        ),
        workspace_dashboard_markdown(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def delete_current_workspace(workspace_id):
    """Delete the selected custom workspace and activate a safe fallback."""
    global current_workspace_id, current_session_id, available_download_files
    if not workspace_id:
        return _workspace_action_noop()
    if not is_initialized or global_chatbot is None:
        input_update, file_update = _composer_updates(enabled=False)
        return (
            gr.update(), [], "请先初始化系统", workspace_dashboard_markdown(),
            gr.update(choices=[], value=None, visible=False),
            gr.update(choices=[], value=None), input_update, file_update,
            gr.update(visible=False), [],
        )

    db = DatabaseManager()
    try:
        deleted = db.delete_workspace(global_chatbot.user_id, str(workspace_id))
        workspaces = db.list_workspaces(global_chatbot.user_id)
        if not workspaces:
            fallback_id = db.ensure_personal_workspace(global_chatbot.user_id)
            workspaces = db.list_workspaces(global_chatbot.user_id)
        else:
            fallback_id = workspaces[0]["workspace_id"]
        session_id = db.get_user_default_session_id(
            global_chatbot.user_id, fallback_id
        )
        choices = _workspace_choices(db, global_chatbot.user_id)
        active_run = db.get_active_background_task(global_chatbot.user_id)
    except Exception as exc:
        db.close()
        active_run = _get_active_background_task(global_chatbot.user_id)
        pairs, _, gallery_update, chart_images, downloads = (
            _session_runtime_view(
                global_chatbot.user_id,
                global_chatbot.workspace_id,
                global_chatbot.session_id,
            )
        )
        input_update, file_update = _composer_updates(active_run)
        return (
            gr.update(), pairs,
            f"删除工作区失败：{str(exc) or type(exc).__name__}",
            workspace_dashboard_markdown(), downloads,
            _session_selector_state(global_chatbot.session_id),
            input_update, file_update, gallery_update, chart_images,
        )
    else:
        db.close()

    current_workspace_id = fallback_id
    current_session_id = session_id
    global_chatbot.workspace_id = fallback_id
    global_chatbot.session_id = session_id
    global_chatbot.last_run_metadata = {}
    global_chatbot._load_history()
    available_download_files = []
    pairs, runtime_status, gallery_update, chart_images, downloads = (
        _session_runtime_view(
            global_chatbot.user_id, fallback_id, session_id
        )
    )
    input_update, file_update = _composer_updates(active_run)
    status = (
        _active_task_status(active_run, session_id)
        if active_run
        else runtime_status
        or ("工作区已删除" if deleted else "工作区已不存在，列表已刷新")
    )
    return (
        gr.update(choices=choices, value=fallback_id),
        pairs,
        status,
        workspace_dashboard_markdown(),
        downloads,
        _session_selector_state(session_id),
        input_update,
        file_update,
        gallery_update,
        chart_images,
    )


def switch_history_session(session_id):
    global current_session_id, available_download_files
    if not is_initialized or global_chatbot is None:
        input_update, file_update = _composer_updates(enabled=False)
        return (
            [], "请先初始化系统", gr.update(), gr.update(visible=False), [],
            gr.update(choices=[], value=None, visible=False), input_update,
            file_update,
        )
    if not session_id or session_id == global_chatbot.session_id:
        active_run = _get_active_background_task(global_chatbot.user_id)
        pairs, runtime_status, gallery_update, chart_images, downloads = (
            _session_runtime_view(
                global_chatbot.user_id,
                global_chatbot.workspace_id,
                global_chatbot.session_id,
            )
        )
        input_update, file_update = _composer_updates(active_run)
        return (
            pairs,
            (
                _active_task_status(active_run, global_chatbot.session_id)
                if active_run
                else runtime_status or f"当前会话 · {global_chatbot.session_id[:8]}…"
            ),
            _session_selector_state(global_chatbot.session_id),
            gallery_update,
            chart_images,
            downloads,
            input_update,
            file_update,
        )
    db = DatabaseManager()
    try:
        if not db.session_belongs_to_workspace(
            global_chatbot.user_id, global_chatbot.workspace_id, session_id
        ):
            raise ValueError("会话不存在或不属于当前工作区")
        active_run = db.get_active_background_task(global_chatbot.user_id)
    finally:
        db.close()
    current_session_id = session_id
    global_chatbot.session_id = session_id
    global_chatbot.last_run_metadata = {}
    global_chatbot._load_history()
    available_download_files = []
    pairs, runtime_status, gallery_update, chart_images, downloads = (
        _session_runtime_view(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            session_id,
        )
    )
    input_update, file_update = _composer_updates(active_run)
    return (
        pairs,
        (
            _active_task_status(active_run, session_id)
            if active_run
            else runtime_status or f"已载入历史会话 · {session_id[:8]}…"
        ),
        _session_selector_state(session_id),
        gallery_update,
        chart_images,
        downloads,
        input_update,
        file_update,
    )


def start_new_session():
    global current_session_id, available_download_files
    if not is_initialized or global_chatbot is None:
        input_update, file_update = _composer_updates(enabled=False)
        return (
            [], "请先初始化系统", gr.update(), gr.update(visible=False), [],
            gr.update(choices=[], value=None, visible=False), input_update,
            file_update,
        )
    db = DatabaseManager()
    try:
        session_id = db.create_workspace_session(
            global_chatbot.user_id, global_chatbot.workspace_id
        )
        active_run = db.get_active_background_task(global_chatbot.user_id)
    finally:
        db.close()
    current_session_id = session_id
    global_chatbot.session_id = session_id
    global_chatbot.chat_history = []
    global_chatbot.last_run_metadata = {}
    available_download_files = []
    input_update, file_update = _composer_updates(active_run)
    return (
        [],
        (
            _active_task_status(active_run, session_id)
            if active_run
            else "已创建新会话"
        ),
        _session_selector_state(session_id),
        gr.update(visible=False),
        [],
        gr.update(choices=[], value=None, visible=False),
        input_update,
        file_update,
    )


def _delete_session_noop():
    """Keep every bound component unchanged when confirmation is cancelled."""
    return tuple(gr.update() for _ in range(9))


def _delete_session_button_state(session_id=None):
    return gr.update(visible=bool(session_id))


def delete_history_session(session_id):
    """Delete a selected completed session and move to a valid next session."""
    global current_session_id, available_download_files
    if not session_id:
        return _delete_session_noop()
    if not is_initialized or global_chatbot is None:
        input_update, file_update = _composer_updates(enabled=False)
        return (
            [], "请先初始化系统", gr.update(choices=[], value=None),
            gr.update(visible=False), [],
            gr.update(choices=[], value=None, visible=False), input_update,
            file_update, gr.update(visible=False),
        )

    db = DatabaseManager()
    try:
        deleted = db.delete_workspace_session(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            str(session_id),
        )
        sessions = db.list_workspace_sessions(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            limit=100,
        )
        if sessions:
            next_session_id = sessions[0]["session_id"]
        else:
            next_session_id = db.create_workspace_session(
                global_chatbot.user_id,
                global_chatbot.workspace_id,
            )
        active_run = db.get_active_background_task(global_chatbot.user_id)
    except Exception as exc:
        db.close()
        active_run = _get_active_background_task(global_chatbot.user_id)
        pairs, _, gallery_update, chart_images, downloads = (
            _session_runtime_view(
                global_chatbot.user_id,
                global_chatbot.workspace_id,
                global_chatbot.session_id,
            )
        )
        input_update, file_update = _composer_updates(active_run)
        return (
            pairs,
            f"删除失败：{str(exc) or type(exc).__name__}",
            _session_selector_state(global_chatbot.session_id),
            gallery_update,
            chart_images,
            downloads,
            input_update,
            file_update,
            gr.update(visible=True),
        )
    else:
        db.close()

    current_session_id = next_session_id
    global_chatbot.session_id = next_session_id
    global_chatbot.last_run_metadata = {}
    global_chatbot._load_history()
    available_download_files = []
    pairs, runtime_status, gallery_update, chart_images, downloads = (
        _session_runtime_view(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            next_session_id,
        )
    )
    input_update, file_update = _composer_updates(active_run)
    status = (
        _active_task_status(active_run, next_session_id)
        if active_run
        else runtime_status
        or ("会话已删除" if deleted else "会话已不存在，列表已刷新")
    )
    return (
        pairs,
        status,
        _session_selector_state(next_session_id),
        gallery_update,
        chart_images,
        downloads,
        input_update,
        file_update,
        gr.update(visible=False),
    )


def _operation_modal_open_state(
    action,
    title,
    description,
    *,
    name_value="",
    name_visible=False,
    name_label="工作区名称",
    confirm_label="确认",
    confirm_variant="primary",
    confirm_interactive=True,
):
    """Build one consistent, in-page modal state for workbench operations."""
    return (
        gr.update(visible=True),
        action,
        f"### {title}",
        description,
        gr.update(
            value=name_value,
            visible=name_visible,
            label=name_label,
            interactive=True,
        ),
        gr.update(
            value=confirm_label,
            variant=confirm_variant,
            interactive=confirm_interactive,
        ),
        gr.update(value="", visible=False),
    )


def open_create_workspace_modal():
    return _operation_modal_open_state(
        "create_workspace",
        "新建工作区",
        "工作区会独立保存记忆、历史会话和任务轨迹。",
        name_visible=True,
        name_label="工作区名称",
        confirm_label="创建工作区",
    )


def _selected_workspace_name(workspace_id):
    if not is_initialized or global_chatbot is None or not workspace_id:
        return None
    db = DatabaseManager()
    try:
        for workspace in db.list_workspaces(global_chatbot.user_id):
            if str(workspace.get("workspace_id")) == str(workspace_id):
                return str(workspace.get("display_name") or "工作区")
    finally:
        db.close()
    return None


def open_rename_workspace_modal(workspace_id):
    try:
        display_name = _selected_workspace_name(workspace_id)
    except Exception as exc:
        return _operation_modal_open_state(
            None,
            "无法读取工作区",
            f"{str(exc) or type(exc).__name__}",
            confirm_label="暂不可用",
            confirm_interactive=False,
        )
    if not display_name:
        return _operation_modal_open_state(
            None,
            "请先选择工作区",
            "选择一个工作区后才能修改名称。",
            confirm_label="暂不可用",
            confirm_interactive=False,
        )
    return _operation_modal_open_state(
        "rename_workspace",
        "重命名工作区",
        f"当前名称：**{display_name}**",
        name_value=display_name,
        name_visible=True,
        name_label="新的工作区名称",
        confirm_label="保存名称",
    )


def open_delete_workspace_modal(workspace_id):
    try:
        display_name = _selected_workspace_name(workspace_id)
    except Exception as exc:
        return _operation_modal_open_state(
            None,
            "无法读取工作区",
            f"{str(exc) or type(exc).__name__}",
            confirm_label="暂不可用",
            confirm_interactive=False,
        )
    if not display_name:
        return _operation_modal_open_state(
            None,
            "请先选择工作区",
            "选择一个工作区后才能删除。",
            confirm_label="暂不可用",
            confirm_interactive=False,
        )
    if global_chatbot and str(workspace_id) == personal_workspace_id(
        global_chatbot.user_id
    ):
        return _operation_modal_open_state(
            None,
            "个人工作区不可删除",
            "个人工作区是账号的默认兜底空间，可以重命名，但不能删除。",
            confirm_label="不可删除",
            confirm_interactive=False,
        )
    return _operation_modal_open_state(
        "delete_workspace",
        "删除工作区",
        (
            f"即将删除 **{display_name}**。其中的工作区记忆和全部会话"
            "将被永久删除，此操作无法撤销。"
        ),
        confirm_label="确认删除",
        confirm_variant="stop",
    )


def open_create_session_modal():
    try:
        display_name = _selected_workspace_name(
            global_chatbot.workspace_id if global_chatbot else None
        )
    except Exception:
        display_name = None
    return _operation_modal_open_state(
        "create_session",
        "新建会话",
        f"将在 **{display_name or '当前工作区'}** 中创建一个新的空会话。",
        confirm_label="创建会话",
    )


def open_delete_session_modal(session_id):
    if not session_id:
        return _operation_modal_open_state(
            None,
            "请先选择会话",
            "选择一条最近会话后才能删除。",
            confirm_label="暂不可用",
            confirm_interactive=False,
        )
    title = "当前会话"
    if is_initialized and global_chatbot is not None:
        db = None
        try:
            db = DatabaseManager()
            sessions = db.list_workspace_sessions(
                global_chatbot.user_id,
                global_chatbot.workspace_id,
                limit=100,
            )
            selected = next(
                (
                    row for row in sessions
                    if str(row.get("session_id")) == str(session_id)
                ),
                None,
            )
            if selected:
                title = str(selected.get("title") or title)
        except Exception:
            pass
        finally:
            if db is not None:
                db.close()
    return _operation_modal_open_state(
        "delete_session",
        "删除会话",
        f"即将删除 **{title}** 及其聊天记录，此操作无法撤销。",
        confirm_label="确认删除",
        confirm_variant="stop",
    )


def close_operation_modal():
    return (
        gr.update(visible=False),
        None,
        gr.update(value=""),
        gr.update(value="", visible=False),
    )


def _session_result_as_workspace_result(result):
    """Map the common eight session outputs into the workspace output shape."""
    return (
        gr.update(),
        result[0],
        result[1],
        gr.update(),
        result[5],
        result[2],
        result[6],
        result[7],
        result[3],
        result[4],
    )


def _operation_modal_error(message, action, display_name):
    return (
        *_workspace_action_noop(),
        gr.update(),
        gr.update(visible=True),
        gr.update(value=display_name),
        action,
        gr.update(value=f"**无法完成操作：** {message}", visible=True),
    )


def execute_operation_modal(
    action, display_name, selected_workspace_id, selected_session_id
):
    """Execute modal-confirmed operations without browser-native dialogs."""
    normalized_name = str(display_name or "").strip()
    if action in {"create_workspace", "rename_workspace"}:
        if not normalized_name:
            return _operation_modal_error(
                "工作区名称不能为空。", action, display_name
            )
        if len(normalized_name) > 100:
            return _operation_modal_error(
                "工作区名称不能超过 100 个字符。", action, display_name
            )

    try:
        if action == "create_workspace":
            workspace_result = create_new_workspace(normalized_name)
            delete_button_update = gr.update(visible=False)
        elif action == "rename_workspace":
            workspace_result = rename_current_workspace(
                selected_workspace_id, normalized_name
            )
            delete_button_update = gr.update()
        elif action == "delete_workspace":
            workspace_result = delete_current_workspace(selected_workspace_id)
            delete_button_update = gr.update(visible=False)
        elif action == "create_session":
            workspace_result = _session_result_as_workspace_result(
                start_new_session()
            )
            delete_button_update = gr.update(visible=False)
        elif action == "delete_session":
            session_result = delete_history_session(selected_session_id)
            workspace_result = _session_result_as_workspace_result(
                session_result[:8]
            )
            delete_button_update = session_result[8]
        else:
            return _operation_modal_error(
                "操作已失效，请关闭弹窗后重试。", action, display_name
            )
    except Exception as exc:
        return _operation_modal_error(
            str(exc) or type(exc).__name__, action, display_name
        )

    operation_status = workspace_result[2]
    if isinstance(operation_status, str) and "失败" in operation_status:
        return _operation_modal_error(
            operation_status, action, display_name
        )

    return (
        *workspace_result,
        delete_button_update,
        gr.update(visible=False),
        gr.update(value=""),
        None,
        gr.update(value="", visible=False),
    )

def _append_progress_text(existing, content, max_chars=MAX_PERSISTED_PROGRESS_CHARS):
    return append_progress_text(
        existing,
        content,
        max_chars=max_chars,
    )


async def _persist_background_progress(job_id, user_id, progress_queue):
    progress_text = "任务已提交，等待后台执行。"
    db = DatabaseManager()
    last_flush = 0.0
    dirty = True
    try:
        while True:
            item = None
            reached_end = False
            try:
                item = await asyncio.to_thread(progress_queue.get, True, 0.4)
                reached_end = item is None
            except queue.Empty:
                pass
            if item is not None:
                progress_text = _append_progress_text(progress_text, item)
                dirty = True
            now = time.monotonic()
            if dirty and (reached_end or now - last_flush >= 0.8):
                db.update_background_task_run(
                    job_id,
                    user_id,
                    progress_text=progress_text,
                )
                dirty = False
                last_flush = now
            if reached_end:
                return progress_text
    finally:
        db.close()


async def _finish_progress_consumer(
    progress_consumer, progress_queue, fallback_text, final_chunk=None
):
    if progress_consumer is None:
        return _append_progress_text(fallback_text, final_chunk)
    if final_chunk:
        progress_queue.put(final_chunk)
    progress_queue.put(None)
    try:
        return await progress_consumer
    except Exception as exc:
        print(f"后台进度持久化异常: {exc}")
        return _append_progress_text(fallback_text, final_chunk)


def _get_background_execution_lock():
    global background_execution_lock
    if background_execution_lock is None:
        background_execution_lock = asyncio.Lock()
    return background_execution_lock


async def _run_background_task(
    job_id,
    username,
    user_id,
    workspace_id,
    session_id,
    query,
):
    progress_queue = queue.Queue()
    progress_consumer = None
    worker = None
    response_text = ""
    output_files = []
    try:
        async with _get_background_execution_lock():
            db = DatabaseManager()
            try:
                db.update_background_task_run(
                    job_id, user_id, status="running"
                )
            finally:
                db.close()
            progress_consumer = asyncio.create_task(
                _persist_background_progress(job_id, user_id, progress_queue)
            )
            progress_queue.put("后台执行器已接管任务，开始运行 Agent。")

            worker = OAI_ChatBot(
                username=username,
                use_multi_agent=True,
                workspace_id=workspace_id,
            )
            worker.session_id = session_id
            worker._load_history()
            result = await worker.process_query(
                query,
                stream_callback=lambda content: progress_queue.put(content),
                user_message_persisted=True,
            )
            response_text = str(result.get("response", "") or "")

        progress_text = await _finish_progress_consumer(
            progress_consumer,
            progress_queue,
            "任务已提交，后台执行已完成。",
        )
        combined_output = f"{progress_text}\n{response_text}".strip()
        output_files = [
            path
            for path in parse_output_files(combined_output)
            if path and os.path.exists(path)
        ]
        _log_user_files(
            output_files,
            file_type="output",
            user_id=user_id,
            session_id=session_id,
        )
        db = DatabaseManager()
        try:
            db.update_background_task_run(
                job_id,
                user_id,
                status="completed",
                progress_text=progress_text,
                response_text=response_text,
                output_files=output_files,
            )
        finally:
            db.close()
    except Exception as exc:
        failure_text = f"后台任务执行失败：{str(exc) or type(exc).__name__}"
        progress_text = await _finish_progress_consumer(
            progress_consumer,
            progress_queue,
            "任务已提交。",
            final_chunk=failure_text,
        )
        db = DatabaseManager()
        try:
            db.save_message(user_id, session_id, "assistant", failure_text)
            db.update_background_task_run(
                job_id,
                user_id,
                status="failed",
                progress_text=progress_text,
                response_text=failure_text,
                error_message=str(exc) or type(exc).__name__,
                output_files=output_files,
            )
        finally:
            db.close()
    finally:
        if worker is not None:
            try:
                worker.db_manager.close()
            except Exception:
                pass


def _schedule_background_task(**job):
    execution_loop = _ensure_execution_loop()
    future = asyncio.run_coroutine_threadsafe(
        _run_background_task(**job), execution_loop
    )
    job_id = job["job_id"]
    with background_futures_lock:
        background_futures[job_id] = future

    def cleanup_finished(done_future):
        with background_futures_lock:
            background_futures.pop(job_id, None)
        try:
            done_future.result()
        except Exception as exc:
            print(f"后台任务 {job_id} 异常结束: {exc}")

    future.add_done_callback(cleanup_finished)


def _session_runtime_view(user_id, workspace_id, session_id):
    db = DatabaseManager()
    try:
        messages = db.load_chat_history(user_id, session_id)
        latest_run = db.get_latest_session_background_task(
            user_id, workspace_id, session_id
        )
        output_files = db.list_session_output_files(user_id, session_id)
    finally:
        db.close()

    pairs = _chatbot_history_pairs(messages)
    status = None
    combined_output = ""
    if latest_run:
        run_status = latest_run.get("status")
        progress_text = str(latest_run.get("progress_text") or "")
        response_text = str(latest_run.get("response_text") or "")
        combined_output = f"{progress_text}\n{response_text}".strip()
        if run_status in {"pending", "running"}:
            label = "排队中" if run_status == "pending" else "后台执行中"
            running_reply = (
                f"⏳ **{label}**\n\n"
                + (progress_text[-12000:] or "等待后台执行器接管任务。")
            )
            query_text = str(latest_run.get("query_text") or "")
            if pairs and pairs[-1][0] == query_text and not pairs[-1][1]:
                pairs[-1] = (pairs[-1][0], running_reply)
            else:
                pairs.append((query_text, running_reply))
            status = f"任务{label} · 会话 {session_id[:8]}…"
        elif run_status == "failed":
            status = f"后台任务失败 · 会话 {session_id[:8]}…"
        elif run_status == "completed":
            status = f"后台任务已完成 · 会话 {session_id[:8]}…"

        if run_status in {"failed", "completed"} and response_text:
            query_text = str(latest_run.get("query_text") or "")
            if pairs and pairs[-1][0] == query_text and not pairs[-1][1]:
                pairs[-1] = (pairs[-1][0], response_text)

        for path in latest_run.get("output_files", []):
            if path not in output_files:
                output_files.append(path)

    output_files = [path for path in output_files if os.path.exists(path)]
    charts = parse_chart_data(combined_output)
    gallery_update, chart_images = load_charts(charts)
    download_update = gr.update(
        visible=bool(output_files),
        choices=output_files,
        value=None,
        label=f"输出文件 ({len(output_files)})",
    )
    return pairs, status, gallery_update, chart_images, download_update


def _hidden_mesh_view(message=None):
    return (
        gr.update(value=None, visible=False),
        gr.update(choices=[], value=[], visible=False),
        None,
        gr.update(
            value=message or "",
            visible=bool(message),
        ),
    )


def _mesh_preview_status(preview):
    if not preview.get("structure_count"):
        return "未选择要显示的三维结构。请在上方勾选至少一个结构。"
    return (
        f"已显示 **{preview['structure_count']}** 个结构 · "
        f"{preview['vertex_count']:,} 个顶点 · "
        f"{preview['triangle_count']:,} 个三角面  \n"
        "已启用平滑点法线与 PBR 实体材质；可用鼠标拖拽旋转、"
        "滚轮缩放、右键拖拽平移。"
    )


def refresh_session_mesh_view(
    session_id,
    selected_labels=None,
    current_manifest_path=None,
    request: gr.Request = None,
):
    """Restore a mesh once and control the short-lived mesh polling timer."""
    client_id = getattr(request, "session_hash", None) or f"session:{session_id}"
    if not is_initialized or global_chatbot is None or not session_id:
        return (*_hidden_mesh_view(), gr.update(active=False))
    db = DatabaseManager()
    try:
        if not db.session_belongs_to_workspace(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            session_id,
        ):
            return (*_hidden_mesh_view(), gr.update(active=False))
        latest_run = db.get_latest_session_background_task(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            session_id,
        )
        output_files = db.list_session_output_files(
            global_chatbot.user_id,
            session_id,
        )
    finally:
        db.close()

    run_status = str((latest_run or {}).get("status") or "")
    if run_status in {"pending", "running"}:
        # A prior result may still be registered in this session.  Hide it and
        # keep only the dedicated mesh poller alive until the new run ends.
        mesh_refresh_tracker.claim(client_id, session_id, None)
        return (*_hidden_mesh_view(), gr.update(active=True))

    combined_output = ""
    if latest_run:
        combined_output = (
            f"{latest_run.get('progress_text') or ''}\n"
            f"{latest_run.get('response_text') or ''}"
        )
        for path in latest_run.get("output_files", []):
            if path not in output_files:
                output_files.insert(0, path)
    manifest_path = find_mesh_manifest(output_files, combined_output)
    if not manifest_path:
        if not mesh_refresh_tracker.claim(client_id, session_id, None):
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(active=False)
        return (*_hidden_mesh_view(), gr.update(active=False))

    if not mesh_refresh_tracker.claim(client_id, session_id, manifest_path):
        # Polling callbacks can overlap and carry stale component state.  The
        # server-side page identity is authoritative and prevents another
        # Model3D value from being sent for the same result.
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(active=False)

    try:
        preview = build_mesh_preview(manifest_path, None)
    except Exception as exc:
        mesh_refresh_tracker.rollback(client_id, session_id, manifest_path)
        return (
            *_hidden_mesh_view(f"三维网格预览加载失败：{exc}"),
            gr.update(active=False),
        )
    return (
        gr.update(value=preview["preview_path"], visible=True),
        gr.update(
            choices=preview["choices"],
            value=preview["selected_labels"],
            visible=True,
        ),
        manifest_path,
        gr.update(value=_mesh_preview_status(preview), visible=True),
        gr.update(active=False),
    )


def update_mesh_structure_view(selected_labels, manifest_path):
    """Rebuild only the lightweight viewer mesh when a structure is toggled."""
    if not manifest_path:
        return gr.update(value=None, visible=False), gr.update(visible=False)
    try:
        preview = build_mesh_preview(manifest_path, selected_labels or [])
    except Exception as exc:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=f"三维网格预览更新失败：{exc}", visible=True),
        )
    return (
        gr.update(
            value=preview["preview_path"],
            visible=bool(preview["preview_path"]),
        ),
        gr.update(value=_mesh_preview_status(preview), visible=True),
    )


def start_background_task(query, history, files=None, selected_session_id=None):
    if not is_initialized or global_chatbot is None:
        input_update, file_update = _composer_updates(enabled=False)
        return (
            history or [], input_update, gr.update(visible=False), [],
            gr.update(visible=False), workspace_dashboard_markdown(),
            "工作台尚未初始化", gr.update(), file_update,
        )
    query = str(query or "").strip()
    user_id = global_chatbot.user_id
    username = global_chatbot.username
    workspace_id = global_chatbot.workspace_id
    session_id = selected_session_id or global_chatbot.session_id
    active_run = None
    with background_submission_lock:
        db = DatabaseManager()
        try:
            active_run = db.get_active_background_task(user_id)
            if active_run is None:
                if not query and not files:
                    input_update, file_update = _composer_updates()
                    return (
                        history or [], input_update, gr.update(visible=False), [],
                        gr.update(), workspace_dashboard_markdown(),
                        "请输入任务目标或上传任务文件", gr.update(), file_update,
                    )
                if not db.session_belongs_to_workspace(
                    user_id, workspace_id, session_id
                ):
                    raise ValueError("当前会话不属于所选工作区")

                file_paths = save_uploaded_files(
                    files, user_id=user_id, session_id=session_id
                )
                if file_paths:
                    file_info = "\n".join(
                        f"[已上传文件: {path}]" for path in file_paths
                    )
                    effective_query = (
                        f"{file_info}\n{query or '自动处理已上传文件'}"
                    )
                else:
                    effective_query = query

                job_id = str(uuid.uuid4())
                db.create_background_task_run(
                    job_id,
                    user_id,
                    workspace_id,
                    session_id,
                    effective_query,
                )
                db.save_message(user_id, session_id, "user", effective_query)
                db.update_background_task_run(
                    job_id,
                    user_id,
                    progress_text="任务已提交，等待后台执行。",
                )
        finally:
            db.close()

    if active_run:
        pairs, _, gallery_update, chart_images, download_update = (
            _session_runtime_view(user_id, workspace_id, session_id)
        )
        input_update, file_update = _composer_updates(active_run, clear=True)
        return (
            pairs,
            input_update,
            gallery_update,
            chart_images,
            download_update,
            workspace_dashboard_markdown(),
            _active_task_status(active_run, session_id),
            _session_selector_state(session_id),
            file_update,
        )

    _schedule_background_task(
        job_id=job_id,
        username=username,
        user_id=user_id,
        workspace_id=workspace_id,
        session_id=session_id,
        query=effective_query,
    )
    pairs, _, gallery_update, chart_images, download_update = (
        _session_runtime_view(user_id, workspace_id, session_id)
    )
    submitted_run = {
        "status": "pending",
        "session_id": session_id,
    }
    input_update, file_update = _composer_updates(submitted_run, clear=True)
    return (
        pairs,
        input_update,
        gallery_update,
        chart_images,
        download_update,
        workspace_dashboard_markdown(),
        f"任务已转入后台 · 会话 {session_id[:8]}…",
        gr.update(),
        file_update,
    )


def poll_current_session(session_id):
    if not is_initialized or global_chatbot is None:
        input_update, file_update = _composer_updates(enabled=False)
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), input_update, file_update,
        )
    db = DatabaseManager()
    try:
        active_run = db.get_active_background_task(global_chatbot.user_id)
        input_update, file_update = _composer_updates(active_run)
        if not session_id:
            return (
                gr.update(),
                (
                    _active_task_status(active_run)
                    if active_run
                    else gr.update()
                ),
                gr.update(), gr.update(), gr.update(), gr.update(),
                input_update, file_update,
            )
        if not db.session_belongs_to_workspace(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            session_id,
        ):
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), input_update, file_update,
            )
    finally:
        db.close()
    pairs, status, gallery_update, chart_images, download_update = (
        _session_runtime_view(
            global_chatbot.user_id,
            global_chatbot.workspace_id,
            session_id,
        )
    )
    return (
        pairs,
        (
            _active_task_status(active_run, session_id)
            if active_run
            else status or gr.update()
        ),
        _session_selector_state(session_id),
        gallery_update,
        chart_images,
        download_update,
        input_update,
        file_update,
    )


def update_download_button(file_path):
    if file_path and os.path.exists(file_path):
        return gr.update(value=file_path, visible=True)
    return gr.update(value=None, visible=False)


def record_latest_task_feedback(outcome_label, notes=""):
    """Record explicit user feedback for the latest completed run."""
    if not is_initialized or global_chatbot is None:
        return "请先完成并初始化一个任务。", workspace_dashboard_markdown()
    try:
        result = global_chatbot.record_latest_task_feedback(
            outcome_label, notes or ""
        )
        return (
            (
                "反馈已记录："
                f"{result['outcome_label']}（trace: {result['trace_id'][:8]}…）"
            ),
            workspace_dashboard_markdown(),
        )
    except Exception as exc:
        return f"反馈记录失败：{exc}", workspace_dashboard_markdown()

# 清除当前会话的聊天记录
def clear_current_record():
    global chat_history, chart_data, current_charts
    chat_history, chart_data, current_charts = [], [], []
    gallery_update, default_images = load_default_image()
    # 保留下载文件列表，不清除
    file_download_update = gr.update(visible=True, 
                                    choices=available_download_files, 
                                    label=f"可用下载文件 ({len(available_download_files)})"),[]
    # 新增：设置 file_input 的更新状态，清除上传的文件
    file_input_update = gr.update(value=None)
    return [], gallery_update, default_images, file_download_update, file_input_update

# 清除用户的所有聊天记录（包括数据库）
def clear_all_history():
    global chat_history, chart_data, current_charts, global_chatbot, available_download_files
    # 清除前端和内存中的记录
    chat_history, chart_data, current_charts = [], [], []
    # 同时清除下载文件列表
    available_download_files = []
    # 清除数据库中的记录
    if current_user:
        db_manager = DatabaseManager()
        success = db_manager.clear_user_messages(current_user)
        db_manager.close()
        if success:
            status = "✅ 所有历史记录已清除"
            # 同时清除chatbot对象内部的历史记录
            if global_chatbot:
                global_chatbot.clear_chat_history()
        else:
            status = "❌ 清除数据库记录失败"
    else:
        status = "❌ 请先登录"
    gallery_update, default_images = load_default_image()
    # 设置 file_download 的更新状态，仍然可见只不过可下载文件数变为0
    file_download_update = gr.update(visible=True, choices=[], label="可用下载文件(0)")
    # 新增：设置 file_input 的更新状态，清除上传的文件
    file_input_update = gr.update(value=None)
    return [], gallery_update, default_images, status, file_download_update, file_input_update

# ================== 登录逻辑 ==================
def login_logic(username, password):
    if not username or not password:
        return "❌ 请输入用户名和密码"
    db_manager = DatabaseManager()
    # 先检查用户是否存在
    if not db_manager.find_user_id(username):
        db_manager.close()
        return "❌ 用户不存在，请先<a href='/register'>注册</a>"
    # 用户存在，验证密码
    if db_manager.validate_user(username, password):
        html = f"""
        <div>
            ✅ 登录成功，欢迎 <b>{username}</b>！正在跳转到聊天页面...
            <meta http-equiv="refresh" content="1; url=/set_session?username={username}">
        </div>
        """
        db_manager.close()
        return html
    else:
        db_manager.close()
        return "❌ 密码错误，请重新输入"

# 放在文件顶部合适位置
SUCCESS_TPL = """
<div style="text-align: center; padding: 20px;">
  <h2 style="color: #4CAF50;">✅ 注册成功！</h2>
  <p style="font-size: 18px; margin-top: 20px;">
    即将在 <span style="font-weight: bold; color: #2196F3; font-size: 24px;">{s}</span> 秒后自动跳转到登录页面…
  </p>
  <p style="margin-top: 20px;">
    如果没有自动跳转，请点击
    <a href="/login" style="color: #2196F3; text-decoration: none; border-bottom: 1px dashed #2196F3;">登录页</a>
  </p>
</div>
"""

# ================== 注册逻辑（流式） ==================
def register_logic(username, password):
    if not username or not password:
        # 第一个输出是 HTML 展示，第二个输出是隐藏“跳转标志”
        yield "❌ 请输入用户名和密码", ""
        return
    db_manager = DatabaseManager()
    success = db_manager.register_user(username, password)
    db_manager.close()
    if not success:
        yield "❌ 注册失败，用户名已存在", ""
        return

    # 流式倒计时
    for s in range(3, 0, -1):
        yield SUCCESS_TPL.format(s=s), ""
        import time; time.sleep(1)

    # 结束：展示 0，并发出跳转信号
    yield SUCCESS_TPL.format(s=0), "go"
AUTH_CSS = """
:root {
    --brand-950: #102a2e;
    --brand-800: #17464b;
    --brand-600: #0f766e;
    --brand-500: #159486;
    --brand-100: #dff5f1;
    --ink-900: #172126;
    --ink-600: #5e6d73;
    --line: #dce6e7;
}
body, .gradio-container {
    font-family: Inter, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif !important;
}
.gradio-container {
    min-height: 100vh !important;
    max-width: none !important;
    background:
        radial-gradient(circle at 12% 16%, rgba(21, 148, 134, .14), transparent 28%),
        radial-gradient(circle at 88% 82%, rgba(45, 110, 121, .11), transparent 30%),
        #f4f8f8 !important;
}
.auth-shell {
    min-height: 100vh;
    align-items: center !important;
    justify-content: center !important;
    padding: 32px 20px;
}
.auth-card {
    width: min(440px, calc(100vw - 32px)) !important;
    flex: 0 1 440px !important;
    padding: 38px 38px 32px !important;
    border: 1px solid rgba(220, 230, 231, .9) !important;
    border-radius: 24px !important;
    background: rgba(255, 255, 255, .94) !important;
    box-shadow: 0 24px 70px rgba(27, 64, 69, .13) !important;
    backdrop-filter: blur(16px);
}
.auth-brand { text-align: center; margin-bottom: 24px; }
.auth-mark {
    display: inline-flex; align-items: center; justify-content: center;
    width: 48px; height: 48px; border-radius: 15px;
    color: white; font-size: 20px; font-weight: 800;
    background: linear-gradient(145deg, var(--brand-800), var(--brand-500));
    box-shadow: 0 10px 26px rgba(15, 118, 110, .24);
}
.auth-brand h1 { margin: 16px 0 6px; color: var(--ink-900); font-size: 25px; }
.auth-brand p { margin: 0; color: var(--ink-600); font-size: 14px; line-height: 1.6; }
.auth-card .form { border: 0 !important; }
.auth-card input {
    border-radius: 12px !important;
    border-color: var(--line) !important;
    background: #fbfdfd !important;
}
.auth-card button.primary, .auth-card button.secondary {
    min-height: 44px; border: 0 !important; border-radius: 12px !important;
    color: white !important;
    background: linear-gradient(135deg, var(--brand-800), var(--brand-600)) !important;
    box-shadow: 0 9px 22px rgba(15, 118, 110, .20) !important;
}
.auth-footer { margin-top: 18px; text-align: center; color: var(--ink-600); font-size: 14px; }
.auth-footer a { color: var(--brand-600); font-weight: 650; text-decoration: none; }
.auth-status { min-height: 12px; color: var(--ink-600); }
"""


def _gradio_major_version():
    try:
        return int(str(gr.__version__).split(".", 1)[0])
    except (AttributeError, TypeError, ValueError):
        return 5


GRADIO_MAJOR_VERSION = _gradio_major_version()


def _blocks_css_kwargs(css):
    """Gradio 4/5 load CSS on Blocks; Gradio 6 loads it when mounted."""
    return {"css": css} if GRADIO_MAJOR_VERSION < 6 else {}


def _mount_gradio_with_css(fastapi_app, blocks, path, css):
    kwargs = {"css": css} if GRADIO_MAJOR_VERSION >= 6 else {}
    return gr.mount_gradio_app(fastapi_app, blocks, path=path, **kwargs)


# 通用布局函数
def make_auth_page(title, btn_text, btn_variant, callback_fn, footer_html):
    with gr.Blocks(title=title, **_blocks_css_kwargs(AUTH_CSS)) as page:
        with gr.Row(elem_classes="auth-shell"):
            with gr.Column(elem_classes="auth-card"):
                gr.HTML(
                    f"""
                    <div class="auth-brand">
                        <div class="auth-mark">KL</div>
                        <h1>{title}</h1>
                        <p>OA 医学影像多智能体研究工作台</p>
                    </div>
                    """
                )
                username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                btn = gr.Button(btn_text, variant=btn_variant)
                status = gr.HTML(elem_classes="auth-status")
                gr.HTML(f'<div class="auth-footer">{footer_html}</div>')
                btn.click(callback_fn, inputs=[username, password], outputs=status)
    return page

# ================== 登录界面 ==================
login_app = make_auth_page(
    title="🔑 用户登录",
    btn_text="登录",
    btn_variant="primary",
    callback_fn=login_logic,
    footer_html="还没有账号？<a href='/register'>立即注册</a>"
)

# ================== 注册界面（流式 + 前端跳转） ==================
with gr.Blocks(
    title="📝 用户注册", **_blocks_css_kwargs(AUTH_CSS)
) as register_app:
    with gr.Row(elem_classes="auth-shell"):
        with gr.Column(elem_classes="auth-card"):
            gr.HTML(
                """
                <div class="auth-brand">
                    <div class="auth-mark">KL</div>
                    <h1>创建研究工作台账号</h1>
                    <p>用于保存任务轨迹、实验结果与研究工件</p>
                </div>
                """
            )
            username = gr.Textbox(label="用户名", placeholder="请输入用户名")
            password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
            btn = gr.Button("注册", variant="secondary")
            status_html = gr.HTML(elem_classes="auth-status")  # 用来显示倒计时
            redirect_flag = gr.Textbox(visible=False) # 隐藏标志位
            gr.HTML("<div class='auth-footer'>已有账号？<a href='/login'>返回登录</a></div>")

            # 点击注册（流式）
            h = btn.click(register_logic, inputs=[username, password], outputs=[status_html, redirect_flag])

            # 流式完成后再读取 flag 并跳转（稳）
            h.then(
                None,  # 不需要后端函数
                inputs=redirect_flag,  # 读到最后一次的 flag 值
                outputs=None,
                js="""
                (v) => {
                  if (v === "go") {
                    const url = "/login";
                    if (window.top) window.top.location.href = url;
                    else window.location.href = url;
                  }
                }
                """
            )
register_app.queue()
# ================== Gradio 聊天界面 ==================
CHAT_CSS = """
:root {
    --brand: #0f766e;
    --brand-dark: #134e4a;
    --canvas: #f3f6f6;
    --surface: #ffffff;
    --surface-soft: #f8faf9;
    --line: #e1e8e7;
    --ink: #172322;
    --muted: #697977;
}
body, .gradio-container {
    font-family: Inter, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif !important;
}
.gradio-container {
    max-width: none !important; min-height: 100vh !important; padding: 0 !important;
    color: var(--ink) !important; background: var(--canvas) !important;
}
#chat-app-shell {
    width: min(1880px, calc(100% - 18px)); margin: 0 auto;
    padding: 8px 0 12px; gap: 8px !important;
}
.app-header {
    display: flex; align-items: center; justify-content: space-between;
    min-height: 50px; padding: 0 3px;
}
.brand-lockup { display: flex; align-items: center; gap: 11px; }
.brand-mark {
    display: grid; place-items: center; width: 38px; height: 38px;
    border-radius: 11px; color: white; font-size: 13px; font-weight: 800;
    background: linear-gradient(145deg, var(--brand-dark), var(--brand));
}
.brand-copy h1 { margin: 0; font-size: 18px; letter-spacing: -.015em; }
.brand-copy p { margin: 3px 0 0; color: var(--muted); font-size: 11px; }
.header-actions { display: flex; align-items: center; gap: 8px; }
.logout-link {
    display: inline-flex; align-items: center; min-height: 38px; padding: 0 13px;
    border: 1px solid var(--line); border-radius: 10px; color: #42504f !important;
    background: white; font-size: 12px; font-weight: 650; text-decoration: none !important;
}
.status-field { min-width: 320px; }
.status-field input {
    min-height: 38px !important; color: #48615e !important;
    border-color: var(--line) !important; background: white !important;
}
.workspace-grid { align-items: stretch !important; gap: 8px !important; min-height: calc(100vh - 72px); }
.workspace-panel {
    min-width: 0; padding: 11px !important; border: 1px solid var(--line) !important;
    border-radius: 14px !important; background: var(--surface) !important;
    box-shadow: 0 5px 20px rgba(31, 61, 58, .045) !important;
}
.history-panel { max-width: 255px; }
.panel-heading { margin: 0 0 8px; }
.panel-heading h2 { margin: 0; font-size: 14px; }
.panel-heading p { margin: 4px 0 0; color: var(--muted); font-size: 11px; line-height: 1.5; }
.section-label {
    margin: 10px 0 6px; color: #7a8886; font-size: 10px;
    font-weight: 750; letter-spacing: .11em; text-transform: uppercase;
}
.creation-actions, .workspace-edit-actions { gap: 6px !important; }
.creation-actions button, .workspace-edit-actions button {
    min-height: 38px !important; white-space: nowrap;
}
.workspace-edit-actions { margin-top: 7px !important; }
.workspace-edit-actions .delete-workspace { margin-top: 0 !important; }
.session-list { max-height: calc(100vh - 276px); overflow-y: auto; scrollbar-width: thin; }
.session-list > div { border: 0 !important; padding: 0 !important; background: transparent !important; }
.session-list label {
    margin: 3px 0 !important; padding: 9px 10px !important; border: 1px solid transparent !important;
    border-radius: 10px !important; color: #445250 !important; background: transparent !important;
    font-size: 11px !important; line-height: 1.35 !important;
}
.session-list label:hover { background: #f1f6f5 !important; }
.session-list label:has(input:checked) {
    border-color: #cce3df !important; color: var(--brand-dark) !important; background: #eaf5f3 !important;
}
.delete-session, .delete-workspace {
    margin-top: 7px !important;
    animation: session-action-in .16s ease-out;
}
.delete-session button, .delete-workspace button {
    min-height: 34px !important; border-color: #efc9c5 !important;
    color: #a23a32 !important; background: #fff8f7 !important;
    font-size: 11px !important; box-shadow: none !important;
}
.delete-session button:hover, .delete-workspace button:hover {
    border-color: #dfaaa4 !important; background: #fff0ee !important;
}
@keyframes session-action-in {
    from { opacity: 0; transform: translateY(-3px); }
    to { opacity: 1; transform: translateY(0); }
}
.conversation-panel { padding-bottom: 10px !important; }
#chat-window {
    min-height: 540px; overflow: hidden; border: 1px solid var(--line) !important;
    border-radius: 13px !important; background: #fbfcfc !important;
}
#chat-window .message { border-radius: 13px !important; box-shadow: none !important; }
.composer-panel {
    margin-top: 7px; padding: 9px !important; border: 1px solid var(--line) !important;
    border-radius: 13px !important; background: var(--surface-soft) !important;
}
.composer-panel textarea { min-height: 82px !important; line-height: 1.55 !important; }
.file-input { min-height: 138px; margin-top: 5px; }
.file-input > div { min-height: 116px !important; }
.insight-panel { min-width: 400px; max-width: 600px; gap: 7px !important; }
#gallery {
    width: 100%; min-height: 320px; border: 1px solid var(--line) !important;
    border-radius: 12px !important; background: var(--surface-soft) !important;
}
#gallery img { object-fit: contain !important; background: white !important; }
.trajectory-overview {
    padding: 9px 10px !important;
    border: 1px solid var(--line) !important; border-radius: 12px !important;
    background: var(--surface-soft) !important; font-size: 11px !important;
}
.trajectory-overview h3 { margin-top: 0 !important; font-size: 13px !important; }
.trajectory-overview table { width: 100%; font-size: 10px; }
.system-capability-accordion { border-color: var(--line) !important; border-radius: 12px !important; }
.system-capability-accordion > .label-wrap { font-size: 12px !important; font-weight: 720 !important; }
.capability-overview {
    padding: 12px; border: 1px solid #cfe1de; border-radius: 12px;
    color: var(--ink); background: linear-gradient(145deg, #f1f8f7 0%, #fbfdfc 72%);
}
.capability-overview-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
.capability-eyebrow { color: #44817b; font-size: 9px; font-weight: 800; letter-spacing: .16em; }
.capability-overview h3 { margin: 3px 0 2px; font-size: 14px; letter-spacing: -.01em; }
.capability-overview p { margin: 0; color: var(--muted); font-size: 10px; line-height: 1.5; }
.capability-live-status, .capability-mini-status {
    display: inline-flex; align-items: center; gap: 5px; white-space: nowrap;
    color: #50615f; font-size: 9px; font-weight: 700;
}
.capability-live-status {
    padding: 5px 7px; border: 1px solid #d5dfde; border-radius: 999px; background: rgba(255,255,255,.82);
}
.capability-live-status i, .capability-mini-status i {
    width: 6px; height: 6px; border-radius: 50%; background: #9aa9a7;
    box-shadow: 0 0 0 2px rgba(154,169,167,.12);
}
.capability-live-status.is-ready i, .capability-mini-status.is-ready i { background: #18a477; box-shadow: 0 0 0 2px rgba(24,164,119,.13); }
.capability-live-status.is-partial i, .capability-mini-status.is-partial i { background: #d68a22; box-shadow: 0 0 0 2px rgba(214,138,34,.13); }
.capability-live-status.is-connecting i, .capability-mini-status.is-connecting i { background: #4b8fa8; box-shadow: 0 0 0 2px rgba(75,143,168,.13); }
.capability-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 10px; }
.capability-metrics > div { padding: 7px 8px; border: 1px solid rgba(207,225,222,.85); border-radius: 9px; background: rgba(255,255,255,.7); }
.capability-metrics strong { display: block; color: var(--brand-dark); font-size: 15px; line-height: 1; }
.capability-metrics span { display: block; margin-top: 3px; color: #71817f; font-size: 9px; }
.capability-flow { display: flex; align-items: center; gap: 4px; margin-top: 9px; overflow-x: auto; color: #4c6461; font-size: 9px; scrollbar-width: none; }
.capability-flow span { padding: 3px 5px; border-radius: 5px; background: #e8f2f0; white-space: nowrap; }
.capability-flow b { color: #8aa09d; font-weight: 600; }
.capability-tabs { margin-top: 2px !important; }
.capability-tabs button { min-height: 34px !important; font-size: 11px !important; }
.capability-scroll { max-height: 520px; overflow: auto; padding-right: 3px; scrollbar-width: thin; }
.capability-list-section { color: var(--ink); }
.capability-list-intro {
    display: flex; align-items: center; justify-content: space-between; gap: 8px;
    margin-bottom: 8px; padding: 8px 9px; border-radius: 9px; background: var(--surface-soft);
}
.capability-list-intro strong { display: block; font-size: 11px; }
.capability-list-intro div span { display: block; margin-top: 2px; color: var(--muted); font-size: 9px; line-height: 1.4; }
.capability-count { flex: 0 0 auto; color: var(--brand-dark); font-size: 9px; font-weight: 750; }
.capability-agent-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; }
.capability-agent-card { min-width: 0; padding: 9px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
.capability-card-heading { display: flex; align-items: flex-start; gap: 7px; }
.capability-agent-mark {
    display: grid; place-items: center; flex: 0 0 29px; width: 29px; height: 29px;
    border-radius: 8px; color: #265f59; background: #e8f3f1; font-size: 9px; font-weight: 800;
}
.capability-agent-mark.stage-entry, .capability-agent-mark.stage-orchestration { color: #4c4b91; background: #eeedfb; }
.capability-agent-mark.stage-validation { color: #8b5e22; background: #fbf0de; }
.capability-agent-mark.stage-reporting, .capability-agent-mark.stage-interaction { color: #586675; background: #eef1f4; }
.capability-card-heading > div { min-width: 0; flex: 1; }
.capability-card-heading h4 { margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; }
.capability-card-heading code, .capability-tool-copy code { display: block; margin-top: 2px; color: #86928f; font-size: 8px; overflow-wrap: anywhere; }
.capability-stage { flex: 0 0 auto; padding: 2px 5px; border-radius: 999px; color: #51706d; background: #edf5f4; font-size: 8px; font-weight: 750; }
.capability-agent-card > p { min-height: 42px; margin: 7px 0; color: #60706e; font-size: 9px; line-height: 1.55; }
.capability-tags { display: flex; flex-wrap: wrap; gap: 3px; }
.capability-tags span { padding: 2px 5px; border: 1px solid #e3eae9; border-radius: 5px; color: #5e6f6d; background: #fafcfc; font-size: 8px; }
.capability-card-footer { margin-top: 7px; padding-top: 6px; border-top: 1px solid #edf1f0; }
.capability-tool-groups { display: grid; gap: 6px; }
.capability-tool-group { border: 1px solid var(--line); border-radius: 9px; background: #fff; overflow: hidden; }
.capability-tool-group > summary { display: flex; align-items: center; justify-content: space-between; padding: 8px 9px; cursor: pointer; background: #f8faf9; list-style-position: inside; }
.capability-tool-group > summary span { font-size: 10px; font-weight: 750; }
.capability-tool-group > summary small { color: #73817f; font-size: 9px; }
.capability-tool-rows { border-top: 1px solid var(--line); }
.capability-tool-row { display: grid; grid-template-columns: 8px minmax(0, 1fr) auto; gap: 6px; align-items: start; padding: 8px 9px; }
.capability-tool-row + .capability-tool-row { border-top: 1px solid #eef2f1; }
.capability-tool-dot { width: 6px; height: 6px; margin-top: 3px; border-radius: 50%; background: #a9b4b2; }
.capability-tool-dot.is-ready { background: #18a477; box-shadow: 0 0 0 2px rgba(24,164,119,.12); }
.capability-tool-dot.is-connecting { background: #4b8fa8; box-shadow: 0 0 0 2px rgba(75,143,168,.12); }
.capability-tool-copy { min-width: 0; }
.capability-tool-copy strong { display: block; font-size: 10px; }
.capability-tool-copy p { margin: 4px 0; color: #657371; font-size: 9px; line-height: 1.45; }
.capability-tool-owners { display: flex; flex-wrap: wrap; align-items: center; gap: 3px; }
.capability-tool-owners b { margin-right: 2px; color: #8a9694; font-size: 8px; font-weight: 650; }
.capability-tool-owners span { padding: 2px 4px; border-radius: 4px; color: #426a66; background: #edf5f4; font-size: 8px; }
.capability-tool-state { margin-top: 1px; padding: 2px 5px; border-radius: 999px; color: #768381; background: #f0f3f2; font-size: 8px; font-weight: 700; white-space: nowrap; }
.capability-tool-state.is-ready { color: #16765b; background: #e6f6f0; }
.capability-tool-state.is-connecting { color: #46788b; background: #eaf3f6; }
button.primary {
    border-color: transparent !important; background: var(--brand-dark) !important;
    box-shadow: none !important;
}
button.secondary {
    border: 1px solid #aab6b5 !important;
    background: #f8faf9 !important;
    color: #374745 !important;
    box-shadow: 0 1px 1px rgba(31, 61, 58, .04) !important;
}
button.secondary:hover {
    border-color: #7f8f8d !important;
    background: #eef3f2 !important;
}
.delete-session button.secondary, .delete-workspace button.secondary {
    border-color: #d5a39e !important;
    color: #97372f !important;
    background: #fff5f3 !important;
}
.delete-session button.secondary:hover,
.delete-workspace button.secondary:hover {
    border-color: #bd7770 !important;
    background: #ffe9e6 !important;
}
.operation-modal-overlay {
    position: fixed !important;
    inset: 0 !important;
    z-index: 9999 !important;
    width: 100vw !important;
    height: 100vh !important;
    padding: 20px !important;
    border: 0 !important;
    background: rgba(15, 32, 34, .48) !important;
    backdrop-filter: blur(4px);
    animation: modal-backdrop-in .16s ease-out;
}
.operation-modal-card {
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%);
    flex: 0 1 460px !important;
    width: min(460px, calc(100vw - 32px)) !important;
    max-width: 460px !important;
    padding: 20px !important;
    gap: 12px !important;
    border: 1px solid #b9c5c4 !important;
    border-radius: 18px !important;
    background: #ffffff !important;
    box-shadow: 0 24px 70px rgba(13, 35, 37, .24) !important;
    animation: modal-card-in .18s ease-out;
}
.operation-modal-title h3 {
    margin: 0 !important;
    color: #183735 !important;
    font-size: 18px !important;
    letter-spacing: -.015em;
}
.operation-modal-description p {
    margin: 0 !important;
    color: #60706e !important;
    font-size: 12px !important;
    line-height: 1.65 !important;
}
.operation-modal-error {
    padding: 9px 10px !important;
    border: 1px solid #e2aaa4 !important;
    border-radius: 10px !important;
    color: #91372f !important;
    background: #fff3f1 !important;
    font-size: 11px !important;
}
.operation-modal-actions { justify-content: flex-end !important; gap: 8px !important; }
.operation-modal-actions button { min-height: 38px !important; }
@keyframes modal-backdrop-in {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes modal-card-in {
    from { opacity: 0; transform: translate(-50%, calc(-50% + 8px)) scale(.985); }
    to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
}
input, textarea { border-radius: 10px !important; border-color: var(--line) !important; }
@media (max-width: 1180px) {
    #chat-app-shell { width: min(1040px, calc(100% - 14px)); }
    .workspace-grid { flex-direction: column !important; }
    .history-panel, .insight-panel { max-width: none; }
    .insight-panel { min-width: 0; }
    .session-list { max-height: 250px; }
    #chat-window { min-height: 500px; }
}
@media (max-width: 700px) {
    #chat-app-shell { width: calc(100% - 10px); padding-top: 8px; }
    .app-header { align-items: flex-start; gap: 8px; }
    .brand-copy p, .status-field { display: none !important; }
    .workspace-panel { padding: 10px !important; border-radius: 13px !important; }
    .capability-agent-grid { grid-template-columns: 1fr; }
}
"""


TASK_INPUT_KEYBOARD_JS = r"""
() => {
    const installTaskInputKeyboard = () => {
        const root = document.getElementById("task-input");
        const textarea = root?.querySelector("textarea");
        if (!textarea || textarea.dataset.agentEnterBehavior === "bound") {
            return;
        }
        textarea.dataset.agentEnterBehavior = "bound";
        textarea.addEventListener("keydown", (event) => {
            if (
                event.key !== "Enter" ||
                event.isComposing ||
                event.keyCode === 229
            ) {
                return;
            }
            if (event.altKey) {
                event.preventDefault();
                event.stopPropagation();
                const start = textarea.selectionStart ?? textarea.value.length;
                const end = textarea.selectionEnd ?? start;
                textarea.setRangeText("\n", start, end, "end");
                textarea.dispatchEvent(
                    new InputEvent("input", {
                        bubbles: true,
                        inputType: "insertLineBreak",
                        data: "\n",
                    })
                );
                return;
            }
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            const submitButton = root.querySelector("button");
            if (submitButton && !submitButton.disabled) {
                submitButton.click();
            }
        }, true);
    };

    installTaskInputKeyboard();
    window.__agentTaskInputObserver?.disconnect();
    window.__agentTaskInputObserver = new MutationObserver(
        installTaskInputKeyboard
    );
    window.__agentTaskInputObserver.observe(document.body, {
        childList: true,
        subtree: true,
    });
}
"""


with gr.Blocks(
    title="KL 影像研究智能体",
    fill_height=True,
    js=TASK_INPUT_KEYBOARD_JS,
    **_blocks_css_kwargs(CHAT_CSS),
) as chat_app:
    with gr.Column(elem_id="chat-app-shell"):
        with gr.Row(elem_classes="app-header"):
            with gr.Column(scale=2, min_width=240):
                gr.HTML(
                    """
                    <div class="brand-lockup">
                        <div class="brand-mark">KL</div>
                        <div class="brand-copy">
                            <h1>OA 影像研究智能体</h1>
                            <p>自动编排、执行与轨迹留存</p>
                        </div>
                    </div>
                    """
                )
            status_text = gr.Textbox(
                value="正在自动连接…", show_label=False, interactive=False,
                elem_classes="status-field", scale=2,
            )
            initialize_btn = gr.Button("重新连接", variant="secondary", scale=0)
            with gr.Column(scale=0, min_width=64):
                gr.HTML(
                    '<div class="header-actions"><a class="logout-link" href="/logout">退出</a></div>'
                )

        with gr.Row(elem_classes="workspace-grid"):
            with gr.Column(scale=2, min_width=210, elem_classes=["workspace-panel", "history-panel"]):
                gr.HTML(
                    '<div class="panel-heading"><h2>工作区与会话</h2><p>记忆、会话和任务轨迹按工作区隔离</p></div>'
                )
                workspace_selector = gr.Dropdown(
                    label="工作区", choices=[], interactive=True,
                    elem_id="workspace-selector",
                )
                with gr.Row(elem_classes="creation-actions"):
                    new_workspace_btn = gr.Button(
                        "＋ 工作区", variant="secondary", scale=1,
                        min_width=90, elem_classes="new-workspace",
                    )
                    new_session_btn = gr.Button(
                        "＋ 会话", variant="primary", scale=1,
                        min_width=90, elem_classes="new-session",
                    )
                with gr.Row(elem_classes="workspace-edit-actions"):
                    rename_workspace_btn = gr.Button(
                        "重命名工作区", variant="secondary", scale=1,
                        min_width=90, elem_classes="rename-workspace",
                    )
                    delete_workspace_btn = gr.Button(
                        "删除工作区", variant="secondary", scale=1,
                        min_width=90, elem_classes="delete-workspace",
                    )
                gr.HTML('<div class="section-label">最近会话</div>')
                history_session_selector = gr.Radio(
                    choices=[], value=None, show_label=False, interactive=True,
                    elem_classes="session-list",
                )
                delete_session_btn = gr.Button(
                    "删除会话", variant="secondary", visible=False,
                    min_width=0, elem_classes="delete-session",
                )

            with gr.Column(scale=7, min_width=500, elem_classes=["workspace-panel", "conversation-panel"]):
                gr.HTML(
                    '<div class="panel-heading"><h2>自动化任务</h2><p>提交目标后，系统会自动选择 Agent、工具与执行步骤</p></div>'
                )
                chatbot = gr.Chatbot(
                    label="任务轨迹与结果", height=560, elem_id="chat-window",
                    show_label=False,
                )
                with gr.Column(elem_classes="composer-panel"):
                    user_input = gr.Textbox(
                        label="任务目标 · Enter 发送",
                        placeholder="输入要完成的任务，按 Enter 发送给 Agent…",
                        lines=3,
                        submit_btn=True,
                        elem_id="task-input",
                    )
                    file_input = gr.Files(
                        label="任务文件（可选）", height=140,
                        file_count="multiple", elem_classes="file-input",
                    )

            with gr.Column(scale=4, min_width=400, elem_classes=["workspace-panel", "insight-panel"]):
                gr.HTML(
                    '<div class="panel-heading"><h2>结果与运行</h2><p>图表、输出工件和工作区轨迹集中展示</p></div>'
                )
                mesh_manifest_state = gr.State(value=None)
                mesh_structure_selector = gr.CheckboxGroup(
                    label="三维结构 · 勾选显示内容",
                    choices=[],
                    value=[],
                    visible=False,
                    interactive=True,
                    elem_id="mesh-structure-selector",
                )
                mesh_model = gr.Model3D(
                    label="膝关节三维网格",
                    display_mode="solid",
                    clear_color=(0.95, 0.97, 0.99, 1.0),
                    height=480,
                    visible=False,
                    interactive=False,
                    elem_id="cartilage-mesh-viewer",
                )
                mesh_view_status = gr.Markdown(
                    "",
                    visible=False,
                    elem_classes="mesh-view-status",
                )
                gallery = gr.Gallery(
                    label="分析图表", elem_id="gallery", columns=1, rows=2,
                    height=520, visible=False, object_fit="contain",
                )
                file_download = gr.Dropdown(
                    label="输出文件", interactive=True, visible=False
                )
                download_btn = gr.DownloadButton(
                    label="下载所选文件", variant="secondary", visible=False
                )
                gr.HTML('<div class="section-label">工作区轨迹</div>')
                workspace_dashboard = gr.Markdown(
                    workspace_dashboard_markdown(), elem_classes="trajectory-overview"
                )
                refresh_dashboard_btn = gr.Button("刷新概览", variant="secondary")
                with gr.Accordion(
                    "系统能力 · Agent / Tool",
                    open=False,
                    elem_classes="system-capability-accordion",
                ):
                    capability_overview = gr.HTML(capability_overview_html)
                    with gr.Tabs(elem_classes="capability-tabs"):
                        with gr.Tab("Agent · 12"):
                            capability_agents = gr.HTML(
                                capability_agents_html,
                                elem_classes="capability-scroll",
                            )
                        with gr.Tab("Tool · 25"):
                            tools_display = gr.HTML(
                                capability_tools_html,
                                elem_classes="capability-scroll",
                            )
                    refresh_tools_btn = gr.Button(
                        "刷新连接状态", variant="secondary"
                    )
                session_poll_timer = gr.Timer(value=1.0, active=True)
                # Kept separate from general task polling because even an
                # empty Model3D update causes Gradio 5.x to recreate its canvas.
                mesh_refresh_timer = gr.Timer(value=1.0, active=False)

        operation_modal_action = gr.State(value=None)
        with gr.Group(
            visible=False,
            elem_id="operation-modal-overlay",
            elem_classes="operation-modal-overlay",
        ) as operation_modal:
            with gr.Column(elem_classes="operation-modal-card"):
                operation_modal_title = gr.Markdown(
                    "### 操作确认", elem_classes="operation-modal-title"
                )
                operation_modal_description = gr.Markdown(
                    "", elem_classes="operation-modal-description"
                )
                operation_modal_name = gr.Textbox(
                    label="工作区名称",
                    placeholder="请输入工作区名称",
                    lines=1,
                    max_lines=1,
                    visible=False,
                )
                operation_modal_error = gr.Markdown(
                    "", visible=False, elem_classes="operation-modal-error"
                )
                with gr.Row(elem_classes="operation-modal-actions"):
                    operation_modal_cancel = gr.Button(
                        "取消", variant="secondary", scale=1, min_width=100
                    )
                    operation_modal_confirm = gr.Button(
                        "确认", variant="primary", scale=1, min_width=100
                    )

    initialization_outputs = [
        status_text, workspace_selector, workspace_dashboard, chatbot,
        history_session_selector, user_input, file_input,
    ]
    mesh_view_outputs = [
        mesh_model,
        mesh_structure_selector,
        mesh_manifest_state,
        mesh_view_status,
        mesh_refresh_timer,
    ]
    initial_load_event = chat_app.load(
        initialize_chatbot, outputs=initialization_outputs
    )
    initial_load_event.then(
        update_tools_display,
        outputs=[capability_overview, capability_agents, tools_display],
    ).then(
        refresh_session_mesh_view,
        inputs=[
            history_session_selector,
            mesh_structure_selector,
            mesh_manifest_state,
        ],
        outputs=mesh_view_outputs,
    )
    initialization_event = initialize_btn.click(
        initialize_chatbot, outputs=initialization_outputs
    )
    initialization_event.then(
        update_tools_display,
        outputs=[capability_overview, capability_agents, tools_display],
    ).then(
        refresh_session_mesh_view,
        inputs=[
            history_session_selector,
            mesh_structure_selector,
            mesh_manifest_state,
        ],
        outputs=mesh_view_outputs,
    )
    workspace_event = workspace_selector.input(
        switch_workspace,
        inputs=[workspace_selector],
        outputs=[
            chatbot, status_text, workspace_dashboard, file_download,
            history_session_selector, user_input, file_input, gallery, gallery,
        ],
    )
    workspace_event.then(
        _delete_session_button_state, outputs=[delete_session_btn]
    )
    workspace_event.then(
        refresh_session_mesh_view,
        inputs=[
            history_session_selector,
            mesh_structure_selector,
            mesh_manifest_state,
        ],
        outputs=mesh_view_outputs,
    )
    workspace_outputs = [
        workspace_selector, chatbot, status_text, workspace_dashboard,
        file_download, history_session_selector, user_input, file_input,
        gallery, gallery,
    ]
    session_outputs = [
        chatbot, status_text, history_session_selector, gallery, gallery,
        file_download, user_input, file_input,
    ]
    history_session_event = history_session_selector.input(
        switch_history_session,
        inputs=[history_session_selector],
        outputs=session_outputs,
    )
    history_session_event.then(
        _delete_session_button_state,
        inputs=[history_session_selector],
        outputs=[delete_session_btn],
    )
    history_session_event.then(
        refresh_session_mesh_view,
        inputs=[
            history_session_selector,
            mesh_structure_selector,
            mesh_manifest_state,
        ],
        outputs=mesh_view_outputs,
    )

    modal_open_outputs = [
        operation_modal,
        operation_modal_action,
        operation_modal_title,
        operation_modal_description,
        operation_modal_name,
        operation_modal_confirm,
        operation_modal_error,
    ]
    new_workspace_btn.click(
        open_create_workspace_modal,
        outputs=modal_open_outputs,
        queue=False,
    )
    rename_workspace_btn.click(
        open_rename_workspace_modal,
        inputs=[workspace_selector],
        outputs=modal_open_outputs,
    )
    delete_workspace_btn.click(
        open_delete_workspace_modal,
        inputs=[workspace_selector],
        outputs=modal_open_outputs,
    )
    new_session_btn.click(
        open_create_session_modal,
        outputs=modal_open_outputs,
    )
    delete_session_btn.click(
        open_delete_session_modal,
        inputs=[history_session_selector],
        outputs=modal_open_outputs,
    )
    operation_modal_cancel.click(
        close_operation_modal,
        outputs=[
            operation_modal,
            operation_modal_action,
            operation_modal_name,
            operation_modal_error,
        ],
        queue=False,
    )
    modal_execute_inputs = [
        operation_modal_action,
        operation_modal_name,
        workspace_selector,
        history_session_selector,
    ]
    modal_execute_outputs = [
        *workspace_outputs,
        delete_session_btn,
        operation_modal,
        operation_modal_name,
        operation_modal_action,
        operation_modal_error,
    ]
    operation_modal_confirm.click(
        execute_operation_modal,
        inputs=modal_execute_inputs,
        outputs=modal_execute_outputs,
    )
    operation_modal_name.submit(
        execute_operation_modal,
        inputs=modal_execute_inputs,
        outputs=modal_execute_outputs,
    )
    refresh_dashboard_btn.click(
        workspace_dashboard_markdown, outputs=[workspace_dashboard]
    )
    refresh_tools_btn.click(
        update_tools_display,
        outputs=[capability_overview, capability_agents, tools_display],
    )
    submit_event = user_input.submit(
        start_background_task,
        inputs=[
            user_input, chatbot, file_input, history_session_selector,
        ],
        outputs=[
            chatbot, user_input, gallery, gallery, file_download,
            workspace_dashboard, status_text, history_session_selector,
            file_input,
        ],
    )
    submit_event.then(
        refresh_session_mesh_view,
        inputs=[
            history_session_selector,
            mesh_structure_selector,
            mesh_manifest_state,
        ],
        outputs=mesh_view_outputs,
    )
    poll_event = session_poll_timer.tick(
        poll_current_session,
        inputs=[history_session_selector],
        outputs=[
            chatbot, status_text, history_session_selector, gallery, gallery,
            file_download, user_input, file_input,
        ],
    )
    mesh_refresh_timer.tick(
        refresh_session_mesh_view,
        inputs=[
            history_session_selector,
            mesh_structure_selector,
            mesh_manifest_state,
        ],
        outputs=mesh_view_outputs,
    )
    # Only an actual user click may rebuild the viewer.  ``change`` also fires
    # for the checkbox updates returned by the one-second polling callback,
    # creating an endless Model3D reload loop and visible flashing.
    mesh_structure_selector.input(
        update_mesh_structure_view,
        inputs=[mesh_structure_selector, mesh_manifest_state],
        outputs=[mesh_model, mesh_view_status],
    )
    file_download.change(
        update_download_button, inputs=[file_download], outputs=[download_btn]
    )

# ================== FastAPI 集成 ==================
from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="xxx")
from fastapi.responses import HTMLResponse
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!doctype html>
    <html lang="zh-CN">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>OA 影像研究智能体</title>
            <style>
                :root {
                    --brand-950: #102a2e;
                    --brand-800: #17464b;
                    --brand-600: #0f766e;
                    --brand-500: #159486;
                    --ink-900: #172126;
                    --ink-600: #5e6d73;
                    --line: #dce6e7;
                }
                * { box-sizing: border-box; }
                body {
                    font-family: Inter, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    margin: 0;
                    padding: 28px;
                    color: var(--ink-900);
                    background:
                        radial-gradient(circle at 14% 18%, rgba(21,148,134,.15), transparent 28%),
                        radial-gradient(circle at 88% 80%, rgba(45,110,121,.12), transparent 31%),
                        #f4f8f8;
                }
                .hero {
                    width: min(1040px, 100%);
                    display: grid;
                    grid-template-columns: 1.25fr .75fr;
                    overflow: hidden;
                    border: 1px solid rgba(220,230,231,.9);
                    border-radius: 28px;
                    background: rgba(255,255,255,.93);
                    box-shadow: 0 28px 90px rgba(27,64,69,.14);
                    backdrop-filter: blur(18px);
                }
                .hero-main { padding: 70px 64px; }
                .brand { display: flex; align-items: center; gap: 12px; color: var(--brand-800); font-weight: 750; }
                .brand-mark {
                    display: grid; place-items: center; width: 42px; height: 42px;
                    border-radius: 13px; color: white; font-weight: 800;
                    background: linear-gradient(145deg, var(--brand-800), var(--brand-500));
                    box-shadow: 0 9px 24px rgba(15,118,110,.24);
                }
                .eyebrow { margin-top: 46px; color: var(--brand-600); font-size: 12px; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
                h1 { max-width: 620px; margin: 14px 0 18px; color: var(--brand-950); font-size: clamp(36px, 5vw, 58px); line-height: 1.08; letter-spacing: -.035em; }
                .lead { max-width: 600px; margin: 0; color: var(--ink-600); font-size: 17px; line-height: 1.8; }
                .btn-container { display: flex; gap: 12px; margin-top: 36px; }
                .btn {
                    display: inline-flex; align-items: center; justify-content: center;
                    min-width: 120px; min-height: 46px; padding: 0 22px;
                    border: 1px solid transparent; border-radius: 12px;
                    font-size: 15px; font-weight: 700;
                    text-decoration: none;
                    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
                }
                .btn:hover { transform: translateY(-2px); }
                .login-btn { color: white; background: linear-gradient(135deg, var(--brand-800), var(--brand-600)); box-shadow: 0 10px 24px rgba(15,118,110,.20); }
                .register-btn { color: var(--brand-800); border-color: var(--line); background: white; }
                .hero-side {
                    display: flex; flex-direction: column; justify-content: center;
                    padding: 48px; color: white;
                    background: linear-gradient(155deg, var(--brand-950), var(--brand-700, #155d60));
                }
                .hero-side h2 { margin: 0 0 22px; font-size: 18px; }
                .feature { display: flex; gap: 12px; padding: 15px 0; border-top: 1px solid rgba(255,255,255,.12); }
                .feature:first-of-type { border-top: 0; }
                .feature-index { color: #79d8ca; font-size: 12px; font-weight: 800; }
                .feature strong { display: block; margin-bottom: 5px; font-size: 14px; }
                .feature span { color: rgba(255,255,255,.66); font-size: 12px; line-height: 1.55; }
                @media (max-width: 800px) {
                    body { padding: 14px; }
                    .hero { grid-template-columns: 1fr; }
                    .hero-main { padding: 42px 30px; }
                    .hero-side { padding: 30px; }
                    .eyebrow { margin-top: 34px; }
                }
            </style>
        </head>
        <body>
            <main class="hero">
                <section class="hero-main">
                    <div class="brand"><span class="brand-mark">KL</span> OA Research Agent</div>
                    <div class="eyebrow">Medical imaging research workspace</div>
                    <h1>让复杂的 KL 分级研究流程更清晰</h1>
                    <p class="lead">统一编排影像分割、特征提取、模型选择与结果追踪，为每一次研究任务保留可复现轨迹。</p>
                    <div class="btn-container">
                        <a class="btn login-btn" href="/login">进入工作台</a>
                        <a class="btn register-btn" href="/register">创建账号</a>
                    </div>
                </section>
                <aside class="hero-side">
                    <h2>研究流程，一处完成</h2>
                    <div class="feature"><span class="feature-index">01</span><div><strong>智能编排</strong><span>根据研究目标选择合适的 Agent 与工具组合。</span></div></div>
                    <div class="feature"><span class="feature-index">02</span><div><strong>实验可追溯</strong><span>统一记录任务、工件、评估指标与策略版本。</span></div></div>
                    <div class="feature"><span class="feature-index">03</span><div><strong>结果可复核</strong><span>通过确定性门禁验证任务是否真正完成。</span></div></div>
                </aside>
            </main>
        </body>
    </html>
    """
# 添加专门的退出登录接口
@app.get("/logout")
def logout_endpoint(request: Request):
    global current_user, is_initialized, global_chatbot, loop, current_workspace_id, current_session_id
    # 清除用户状态
    current_user = None
    is_initialized = False
    current_workspace_id = None
    current_session_id = None
    # 释放当前界面实例。进程级执行循环必须继续运行，确保用户关闭
    # 页面或退出登录后，已经提交的后台任务仍可完成并写回原会话。
    if global_chatbot:
        global_chatbot = None
    # 清除session并重定向到登录页面
    request.session.pop("username", None)
    return RedirectResponse(url="/login")

# 设置 session 的接口（登录成功后跳转到这里）
@app.get("/set_session")
def set_session(request: Request, username: str):
    global current_user
    current_user = username  # 🔑 修复点
    request.session["username"] = username
    return RedirectResponse(url="/chat")

# 聊天页面访问控制
@app.get("/chat")
def check_login(request: Request):
    if "username" not in request.session:
        return RedirectResponse(url="/login")
    return RedirectResponse(url="/chat_ui")

# 挂载 Gradio 应用
app = _mount_gradio_with_css(app, login_app, path="/login", css=AUTH_CSS)
app = _mount_gradio_with_css(
    app, register_app, path="/register", css=AUTH_CSS
)
app = _mount_gradio_with_css(
    app, chat_app, path="/chat_ui", css=CHAT_CSS
)  # 真正的聊天 UI

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
