import json
import pymysql
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import uuid
import hashlib
import threading
import re

from .trajectory import (
    TRAJECTORY_DATA_PARTITIONS,
    assign_trajectory_partition,
    current_agent_version,
    current_policy_id,
    personal_workspace_id,
    sanitize_for_audit,
    stable_json_hash,
)
from .workspace_memory import normalize_workspace_memory


def derive_conversation_title(content: str, max_length: int = 36) -> str:
    """Build a compact session title from the first meaningful user line."""
    for raw_line in str(content or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or (line.startswith("[已上传文件:") and line.endswith("]")):
            continue
        if len(line) <= max_length:
            return line
        return f"{line[:max_length].rstrip()}…"
    return "新会话"

# ===================== 请求上下文（用于日志记录） =====================
_request_context = threading.local()


def set_request_context(
    user_id: int = None,
    session_id: str = None,
    agent_name: str = None,
    trace_id: str = None,
    workspace_id: str = None,
    agent_version: str = None,
    data_partition: str = None,
):
    """设置当前请求的上下文信息。trace_id 如未提供则自动生成。"""
    if user_id is not None:
        _request_context.user_id = user_id
    if session_id is not None:
        _request_context.session_id = session_id
    if agent_name is not None:
        _request_context.agent_name = agent_name
    if trace_id is not None:
        _request_context.trace_id = trace_id
    elif not hasattr(_request_context, 'trace_id') or not _request_context.trace_id:
        _request_context.trace_id = str(uuid.uuid4())
    if workspace_id is not None:
        _request_context.workspace_id = workspace_id
    elif user_id is not None and not getattr(_request_context, 'workspace_id', None):
        _request_context.workspace_id = personal_workspace_id(user_id)
    if agent_version is not None:
        _request_context.agent_version = agent_version
    elif not getattr(_request_context, 'agent_version', None):
        _request_context.agent_version = current_agent_version()
    if data_partition is not None:
        if data_partition not in TRAJECTORY_DATA_PARTITIONS:
            raise ValueError("invalid trajectory data_partition")
        _request_context.data_partition = data_partition


def get_request_context() -> dict:
    """获取当前请求的上下文信息"""
    return {
        'user_id': getattr(_request_context, 'user_id', None),
        'session_id': getattr(_request_context, 'session_id', None),
        'agent_name': getattr(_request_context, 'agent_name', None),
        'trace_id': getattr(_request_context, 'trace_id', None),
        'workspace_id': getattr(_request_context, 'workspace_id', None),
        'agent_version': getattr(_request_context, 'agent_version', None),
        'data_partition': getattr(_request_context, 'data_partition', None),
    }


def get_trace_id() -> str:
    """获取或自动生成当前 trace_id。保证每次请求都有一个唯一标识。"""
    trace_id = getattr(_request_context, 'trace_id', None)
    if not trace_id:
        trace_id = str(uuid.uuid4())
        _request_context.trace_id = trace_id
    return trace_id


def set_agent_context(agent_name: str):
    """设置当前执行的 Agent 名称"""
    _request_context.agent_name = agent_name


class DatabaseManager:
    _initialized_schemas = set()
    _schema_lock = threading.Lock()

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db_config.json')

        with open(config_path, 'r') as f:
            config = json.load(f)

        self.connection = pymysql.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database=config['database'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        schema_key = os.path.abspath(config_path)
        if schema_key not in type(self)._initialized_schemas:
            with type(self)._schema_lock:
                if schema_key not in type(self)._initialized_schemas:
                    self._create_tables()
                    type(self)._initialized_schemas.add(schema_key)

    def _create_tables(self):
        """创建必要的数据库表"""
        with self.connection.cursor() as cursor:
            cursor.execute('''
               CREATE TABLE IF NOT EXISTS users
               (
                   id INT AUTO_INCREMENT PRIMARY KEY,
                   username VARCHAR(50) UNIQUE NOT NULL,
                   password VARCHAR(255) NOT NULL,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                   ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
               ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workspaces (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    workspace_id VARCHAR(100) NOT NULL,
                    display_name VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_user_workspace (user_id, workspace_id),
                    INDEX idx_workspace_user_time (user_id, created_at),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workspace_memories (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    workspace_id VARCHAR(100) NOT NULL,
                    memory_type VARCHAR(30) NOT NULL,
                    content TEXT NOT NULL,
                    source VARCHAR(30) NOT NULL DEFAULT 'human',
                    active TINYINT(1) NOT NULL DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_workspace_memory_active
                        (user_id, workspace_id, active, updated_at),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workspace_memory_revisions (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    memory_id BIGINT NOT NULL,
                    action VARCHAR(20) NOT NULL,
                    memory_type VARCHAR(30) NOT NULL,
                    content TEXT NOT NULL,
                    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_memory_revision (memory_id, changed_at),
                    FOREIGN KEY (memory_id) REFERENCES workspace_memories(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    workspace_id VARCHAR(100),
                    session_id VARCHAR(100) NOT NULL,
                    title VARCHAR(160),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    conversation_id INT NOT NULL,
                    role ENUM('user', 'assistant','tool','system') NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # ========== 新增表 ==========

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tool_calls_log (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    session_id VARCHAR(100) NOT NULL,
                    agent_name VARCHAR(50) NOT NULL,
                    tool_name VARCHAR(100) NOT NULL,
                    tool_arguments JSON,
                    execution_time_ms INT,
                    status ENUM('success', 'error') NOT NULL DEFAULT 'success',
                    error_message TEXT,
                    result_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_tool_user_time (user_id, created_at),
                    INDEX idx_tool_name_status (tool_name, status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_routing_log (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    session_id VARCHAR(100) NOT NULL,
                    user_query TEXT,
                    routed_agent VARCHAR(50),
                    confidence FLOAT,
                    reasoning TEXT,
                    final_output_agent VARCHAR(50),
                    total_latency_ms INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_routing_user_time (user_id, created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_files (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    session_id VARCHAR(100),
                    file_name VARCHAR(255) NOT NULL,
                    file_path VARCHAR(500) NOT NULL,
                    file_size BIGINT,
                    file_type ENUM('upload', 'output') NOT NULL,
                    source_tool VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user_files_time (user_id, created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    session_id VARCHAR(100) NOT NULL,
                    workspace_id VARCHAR(100),
                    agent_version VARCHAR(100),
                    data_partition VARCHAR(20),
                    task_description TEXT,
                    agent_name VARCHAR(50) DEFAULT 'ORCHESTRATOR_AGENT',
                    status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
                    total_steps INT DEFAULT 0,
                    completed_steps INT DEFAULT 0,
                    result_summary TEXT,
                    error_message TEXT,
                    started_at TIMESTAMP NULL,
                    completed_at TIMESTAMP NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_task_user_time (user_id, created_at),
                    INDEX idx_task_session (session_id, created_at),
                    INDEX idx_task_learning_scope
                        (user_id, workspace_id, agent_version, data_partition, created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS background_task_runs (
                    job_id CHAR(36) PRIMARY KEY,
                    user_id INT NOT NULL,
                    workspace_id VARCHAR(100) NOT NULL,
                    session_id VARCHAR(100) NOT NULL,
                    query_text TEXT NOT NULL,
                    status ENUM('pending', 'running', 'completed', 'failed')
                        NOT NULL DEFAULT 'pending',
                    progress_text MEDIUMTEXT,
                    response_text MEDIUMTEXT,
                    error_message TEXT,
                    output_files_json JSON,
                    started_at TIMESTAMP NULL,
                    completed_at TIMESTAMP NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_background_session
                        (user_id, workspace_id, session_id, created_at),
                    INDEX idx_background_user_status
                        (user_id, status, created_at),
                    INDEX idx_background_status (status, updated_at),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    conversation_id INT NOT NULL,
                    summary_content JSON NOT NULL,
                    summarized_message_count INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # ========== 新增表: Agent 执行日志 ==========

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_execution_log (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    trace_id CHAR(36),
                    user_id INT NOT NULL,
                    session_id VARCHAR(100) NOT NULL,
                    agent_name VARCHAR(50) NOT NULL,
                    total_latency_ms INT,
                    llm_latency_ms INT,
                    tool_count INT DEFAULT 0,
                    success TINYINT(1) NOT NULL DEFAULT 1,
                    error_category VARCHAR(50),
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_agent_trace (trace_id),
                    INDEX idx_agent_name_time (agent_name, created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # ========== 新增表: 编排任务步骤明细 ==========

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_steps (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    task_id BIGINT NOT NULL,
                    trace_id CHAR(36),
                    step_index INT NOT NULL,
                    agent_name VARCHAR(50),
                    task_description TEXT,
                    status ENUM('success', 'failed', 'skipped') DEFAULT 'success',
                    error_category VARCHAR(50),
                    error_message TEXT,
                    output_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_step_task (task_id),
                    INDEX idx_step_trace (trace_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # ========== 新增表: 实验结果（分类指标结构化存储） ==========

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    trace_id CHAR(36),
                    user_id INT,
                    session_id VARCHAR(100),
                    output_dir VARCHAR(500) NOT NULL,
                    suffix VARCHAR(50) NOT NULL DEFAULT '',
                    feature_type VARCHAR(20),
                    classification_method VARCHAR(20),
                    train_path VARCHAR(500),
                    test_path VARCHAR(500),
                    train_sha256 CHAR(64),
                    test_sha256 CHAR(64),
                    train_rows INT,
                    train_unique_patients INT,
                    test_rows INT,
                    test_unique_patients INT,
                    NumPatients INT,
                    AUC DECIMAL(8,6),
                    PR_AUC DECIMAL(8,6),
                    Accuracy DECIMAL(8,6),
                    `Precision` DECIMAL(8,6),
                    Sensitivity DECIMAL(8,6),
                    Specificity DECIMAL(8,6),
                    F1 DECIMAL(8,6),
                    TN INT, FP INT, FN INT, TP INT,
                    selected_feature_count INT,
                    final_c DECIMAL(12,8),
                    random_seed INT,
                    python_version VARCHAR(20),
                    numpy_version VARCHAR(20),
                    pandas_version VARCHAR(20),
                    sklearn_version VARCHAR(20),
                    manifest_sha256 CHAR(64),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_exp_dir_suffix (output_dir(255), suffix),
                    INDEX idx_exp_trace (trace_id),
                    INDEX idx_exp_method (classification_method, feature_type),
                    INDEX idx_exp_auc (AUC)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # Versioned policy registry. Policies are promoted explicitly; trajectory
            # data references them but Agents never mutate active policy state.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS policy_versions (
                    policy_id VARCHAR(100) PRIMARY KEY,
                    policy_type VARCHAR(50) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    status ENUM('draft', 'shadow', 'active', 'retired', 'rolled_back')
                        NOT NULL DEFAULT 'draft',
                    config_json JSON NOT NULL,
                    config_sha256 CHAR(64) NOT NULL,
                    parent_policy_id VARCHAR(100),
                    created_by VARCHAR(100) NOT NULL,
                    approved_by VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    activated_at TIMESTAMP NULL,
                    INDEX idx_policy_type_status (policy_type, status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # Append-only normalized trajectory events. Generic payloads are
            # sanitized before insertion by db_utils.trajectory.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trajectory_events (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    event_id CHAR(36) NOT NULL,
                    schema_version VARCHAR(20) NOT NULL,
                    trace_id CHAR(36) NOT NULL,
                    task_id BIGINT,
                    step_id VARCHAR(100),
                    parent_event_id CHAR(36),
                    event_type VARCHAR(50) NOT NULL,
                    actor_type VARCHAR(30) NOT NULL,
                    actor_name VARCHAR(100) NOT NULL,
                    policy_id VARCHAR(100),
                    user_id INT,
                    workspace_id VARCHAR(100),
                    agent_version VARCHAR(100),
                    data_partition VARCHAR(20),
                    status VARCHAR(20) NOT NULL,
                    duration_ms INT,
                    payload_json JSON NOT NULL,
                    occurred_at VARCHAR(40) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_trajectory_event (event_id),
                    INDEX idx_trajectory_trace (trace_id, created_at),
                    INDEX idx_trajectory_type (event_type, created_at),
                    INDEX idx_trajectory_policy (policy_id, created_at),
                    INDEX idx_trajectory_learning_scope
                        (user_id, workspace_id, agent_version, data_partition, created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # One immutable row per attempted pipeline candidate.  Unlike the
            # task-level trajectory summary, this table preserves paired
            # within-dataset evidence for future offline optimization.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pipeline_trials (
                    trial_id CHAR(36) PRIMARY KEY,
                    schema_version VARCHAR(20) NOT NULL,
                    trace_id CHAR(36) NOT NULL,
                    task_id BIGINT,
                    task_fingerprint CHAR(64),
                    dataset_hash CHAR(64),
                    candidate_id VARCHAR(100) NOT NULL,
                    feature_type VARCHAR(30) NOT NULL,
                    classification_method VARCHAR(30) NOT NULL,
                    model_recipe VARCHAR(50) NOT NULL,
                    candidate_config_sha256 CHAR(64) NOT NULL,
                    policy_id VARCHAR(100),
                    status ENUM('success', 'failed') NOT NULL,
                    validation_scope VARCHAR(50) NOT NULL,
                    test_data_used TINYINT(1) NOT NULL DEFAULT 0,
                    evaluator_version VARCHAR(50),
                    train_sha256 CHAR(64),
                    patient_count INT,
                    outer_folds INT,
                    validation_score DOUBLE,
                    auc DOUBLE,
                    pr_auc DOUBLE,
                    f1 DOUBLE,
                    auc_ci_lower DOUBLE,
                    auc_ci_upper DOUBLE,
                    selection_stability DOUBLE,
                    duration_ms INT,
                    recommended_params_json JSON NOT NULL,
                    metrics_json JSON NOT NULL,
                    metric_intervals_json JSON NOT NULL,
                    details_json JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_pipeline_trial_trace (trace_id, created_at),
                    INDEX idx_pipeline_trial_context
                        (task_fingerprint, candidate_id, status, created_at),
                    INDEX idx_pipeline_trial_dataset
                        (dataset_hash, candidate_id, status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS artifact_registry (
                    artifact_id VARCHAR(40) PRIMARY KEY,
                    trace_id CHAR(36),
                    task_id BIGINT,
                    artifact_type VARCHAR(50) NOT NULL,
                    uri VARCHAR(1000) NOT NULL,
                    sha256 CHAR(64) NOT NULL,
                    created_by VARCHAR(100) NOT NULL,
                    source_artifact_ids JSON NOT NULL,
                    metadata_json JSON NOT NULL,
                    validation_status VARCHAR(30) NOT NULL DEFAULT 'unverified',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_artifact_trace (trace_id, created_at),
                    INDEX idx_artifact_sha (sha256)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    evaluation_id CHAR(36) NOT NULL,
                    trace_id CHAR(36),
                    task_id BIGINT,
                    artifact_id VARCHAR(40),
                    policy_id VARCHAR(100),
                    task_type VARCHAR(50) NOT NULL,
                    method_name VARCHAR(100) NOT NULL,
                    split_type VARCHAR(30) NOT NULL,
                    dataset_hash CHAR(64),
                    evaluator_name VARCHAR(100) NOT NULL,
                    evaluator_version VARCHAR(50) NOT NULL,
                    record_type ENUM('metric', 'gate') NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DOUBLE,
                    gate_status ENUM('pass', 'fail', 'unknown') NULL,
                    details_json JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_evaluation_id (evaluation_id),
                    INDEX idx_eval_trace (trace_id, created_at),
                    INDEX idx_eval_compare
                        (task_type, split_type, metric_name, policy_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # Correct routing quality requires an independent reference label.
            # Execution consistency is kept separately and is not treated as truth.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS routing_feedback (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    routing_log_id BIGINT,
                    trace_id CHAR(36) NOT NULL,
                    expected_agent VARCHAR(50) NOT NULL,
                    observed_agent VARCHAR(50),
                    label_source ENUM('human', 'rule', 'adjudicated') NOT NULL,
                    reviewer_id VARCHAR(100),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_routing_feedback_trace (trace_id),
                    INDEX idx_routing_feedback_agent (expected_agent, created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # Task-level feedback is kept separate from routing reference
            # labels.  Deterministic gates may write ``rule`` feedback, while
            # the UI/API may add an independent human judgement for the same
            # trace without overwriting it.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_feedback (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    trace_id CHAR(36) NOT NULL,
                    task_id BIGINT,
                    feedback_source ENUM('human', 'rule', 'inferred', 'adjudicated')
                        NOT NULL,
                    outcome_label ENUM('correct', 'partial', 'incorrect', 'unknown')
                        NOT NULL,
                    accepted TINYINT(1) NULL,
                    correction_required TINYINT(1) NULL,
                    confidence FLOAT NULL,
                    expected_agent VARCHAR(50),
                    expected_pipeline_id VARCHAR(100),
                    error_types JSON NOT NULL,
                    reviewer_id VARCHAR(100),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_task_feedback_trace_source
                        (trace_id, feedback_source),
                    INDEX idx_task_feedback_outcome
                        (outcome_label, created_at),
                    INDEX idx_task_feedback_task (task_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # ========== 补列: 为已有表增加 trace_id 等字段（幂等） ==========

            _alter_statements = [
                "ALTER TABLE tool_calls_log ADD COLUMN trace_id CHAR(36) NULL",
                "ALTER TABLE tool_calls_log ADD COLUMN error_category VARCHAR(50) NULL",
                "ALTER TABLE tool_calls_log ADD INDEX idx_tool_trace (trace_id)",
                "ALTER TABLE tool_calls_log ADD COLUMN policy_id VARCHAR(100) NULL",
                "ALTER TABLE agent_routing_log ADD COLUMN trace_id CHAR(36) NULL",
                "ALTER TABLE agent_routing_log ADD COLUMN policy_id VARCHAR(100) NULL",
                "ALTER TABLE agent_routing_log ADD COLUMN behavior_probability FLOAT NULL",
                "ALTER TABLE agent_routing_log ADD INDEX idx_routing_trace (trace_id)",
                "ALTER TABLE agent_execution_log ADD COLUMN policy_id VARCHAR(100) NULL",
                "ALTER TABLE agent_execution_log ADD COLUMN model_name VARCHAR(100) NULL",
                "ALTER TABLE agent_execution_log ADD COLUMN prompt_hash CHAR(64) NULL",
                "ALTER TABLE agent_execution_log ADD COLUMN input_tokens INT NULL",
                "ALTER TABLE agent_execution_log ADD COLUMN output_tokens INT NULL",
                "ALTER TABLE agent_execution_log ADD COLUMN estimated_cost DECIMAL(12,6) NULL",
                "ALTER TABLE tasks ADD COLUMN trace_id CHAR(36) NULL",
                "ALTER TABLE tasks ADD COLUMN error_category VARCHAR(50) NULL",
                "ALTER TABLE tasks ADD COLUMN plan_json JSON NULL",
                "ALTER TABLE tasks ADD COLUMN routed_agent VARCHAR(50) NULL",
                "ALTER TABLE tasks ADD COLUMN policy_id VARCHAR(100) NULL",
                "ALTER TABLE tasks ADD COLUMN task_fingerprint CHAR(64) NULL",
                "ALTER TABLE tasks ADD COLUMN execution_fingerprint CHAR(64) NULL",
                "ALTER TABLE tasks ADD COLUMN outcome_status VARCHAR(30) NULL",
                "ALTER TABLE tasks ADD COLUMN completion_gate_status VARCHAR(20) NULL",
                "ALTER TABLE tasks ADD COLUMN outcome_summary_json JSON NULL",
                "ALTER TABLE tasks ADD COLUMN workspace_id VARCHAR(100) NULL",
                "ALTER TABLE tasks ADD COLUMN agent_version VARCHAR(100) NULL",
                "ALTER TABLE tasks ADD COLUMN data_partition VARCHAR(20) NULL",
                "ALTER TABLE tasks ADD INDEX idx_task_trace (trace_id)",
                "ALTER TABLE tasks ADD INDEX idx_task_fingerprint (task_fingerprint, created_at)",
                "ALTER TABLE tasks ADD INDEX idx_task_learning_scope (user_id, workspace_id, agent_version, data_partition, created_at)",
                "ALTER TABLE trajectory_events ADD COLUMN user_id INT NULL",
                "ALTER TABLE trajectory_events ADD COLUMN workspace_id VARCHAR(100) NULL",
                "ALTER TABLE trajectory_events ADD COLUMN agent_version VARCHAR(100) NULL",
                "ALTER TABLE trajectory_events ADD COLUMN data_partition VARCHAR(20) NULL",
                "ALTER TABLE trajectory_events ADD INDEX idx_trajectory_learning_scope (user_id, workspace_id, agent_version, data_partition, created_at)",
                "ALTER TABLE conversations ADD COLUMN workspace_id VARCHAR(100) NULL AFTER user_id",
                "ALTER TABLE conversations ADD COLUMN title VARCHAR(160) NULL AFTER session_id",
                "ALTER TABLE conversations ADD INDEX idx_conversation_workspace (user_id, workspace_id, created_at)",
                "ALTER TABLE conversations ADD INDEX idx_conversation_activity (user_id, workspace_id, updated_at)",
                "ALTER TABLE background_task_runs ADD INDEX idx_background_user_status (user_id, status, created_at)",
                "ALTER TABLE evaluation_results ADD COLUMN method_name VARCHAR(100) NOT NULL DEFAULT 'unknown' AFTER task_type",
            ]
            for stmt in _alter_statements:
                try:
                    cursor.execute(stmt)
                except Exception:
                    pass  # 列/索引已存在则跳过

            # Preserve historical trajectories without pretending their code
            # version was known. Partitioning uses the same stable hash rule as
            # new runs, so repeated task fingerprints cannot cross data splits.
            try:
                cursor.execute(
                    '''INSERT IGNORE INTO workspaces
                       (user_id, workspace_id, display_name)
                       SELECT id,
                              CONCAT(
                                  'personal-',
                                  LEFT(
                                      SHA2(
                                          CONCAT(CHAR(34), CAST(id AS CHAR), CHAR(34)),
                                          256
                                      ),
                                      16
                                  )
                              ),
                              '个人工作区'
                       FROM users'''
                )
                cursor.execute(
                    '''UPDATE conversations
                       SET workspace_id = CONCAT(
                           'personal-',
                           LEFT(
                               SHA2(
                                   CONCAT(CHAR(34), CAST(user_id AS CHAR), CHAR(34)),
                                   256
                               ),
                               16
                           )
                       )
                       WHERE workspace_id IS NULL'''
                )
                cursor.execute(
                    '''UPDATE tasks
                       SET workspace_id = COALESCE(
                               workspace_id,
                               CONCAT(
                                   'personal-',
                                   LEFT(
                                       SHA2(
                                           CONCAT(CHAR(34), CAST(user_id AS CHAR), CHAR(34)),
                                           256
                                       ),
                                       16
                                   )
                               )
                           ),
                           agent_version = COALESCE(agent_version, 'legacy-unversioned'),
                           data_partition = COALESCE(
                               data_partition,
                               CASE
                                   WHEN MOD(CONV(LEFT(SHA2(COALESCE(task_fingerprint, trace_id), 256), 8), 16, 10), 100) < 80 THEN 'train'
                                   WHEN MOD(CONV(LEFT(SHA2(COALESCE(task_fingerprint, trace_id), 256), 8), 16, 10), 100) < 90 THEN 'validation'
                                   ELSE 'test'
                               END
                           )
                       WHERE workspace_id IS NULL
                          OR agent_version IS NULL
                          OR data_partition IS NULL'''
                )
                cursor.execute(
                    '''UPDATE trajectory_events e
                       JOIN tasks t ON t.trace_id = e.trace_id
                       SET e.user_id = COALESCE(e.user_id, t.user_id),
                           e.workspace_id = COALESCE(e.workspace_id, t.workspace_id),
                           e.agent_version = COALESCE(e.agent_version, t.agent_version),
                           e.data_partition = COALESCE(e.data_partition, t.data_partition)
                       WHERE e.user_id IS NULL
                          OR e.workspace_id IS NULL
                          OR e.agent_version IS NULL
                          OR e.data_partition IS NULL'''
                )
            except Exception:
                pass

            # 补索引
            try:
                cursor.execute(
                    'ALTER TABLE conversations ADD INDEX idx_user_session (user_id, session_id)')
            except Exception:
                pass
            try:
                cursor.execute(
                    'ALTER TABLE messages ADD INDEX idx_conv_created (conversation_id, created_at)')
            except Exception:
                pass

            baseline_config = {
                "source": "MULTI_AGENT_POLICY_ID",
                "trajectory_schema": "1.1",
                "automatic_self_modification": False,
            }
            cursor.execute(
                '''INSERT IGNORE INTO policy_versions
                   (policy_id, policy_type, version, status, config_json,
                    config_sha256, created_by, approved_by, activated_at)
                   VALUES (%s, 'system', '1', 'active', %s, %s,
                           'system_bootstrap', 'preexisting_system', NOW())''',
                (
                    current_policy_id(),
                    json.dumps(baseline_config, ensure_ascii=False),
                    stable_json_hash(baseline_config),
                ),
            )

            self.connection.commit()

    # ===================== 原有方法 =====================

    def ensure_workspace(
        self,
        user_id: int,
        workspace_id: str,
        display_name: str = None,
    ) -> str:
        """Register a logical workspace owned by one authenticated user."""
        workspace_id = str(workspace_id or "").strip()
        if not workspace_id or len(workspace_id) > 100:
            raise ValueError("workspace_id must be 1-100 characters")
        display_name = str(display_name or workspace_id).strip()
        if not display_name or len(display_name) > 100:
            raise ValueError("workspace display_name must be 1-100 characters")
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT INTO workspaces (user_id, workspace_id, display_name)
                   VALUES (%s, %s, %s)
                   ON DUPLICATE KEY UPDATE display_name = display_name''',
                (user_id, workspace_id, display_name),
            )
            self.connection.commit()
        return workspace_id

    def ensure_personal_workspace(self, user_id: int) -> str:
        workspace_id = personal_workspace_id(user_id)
        self.ensure_workspace(user_id, workspace_id, "个人工作区")
        self._merge_personal_workspace_aliases(user_id, workspace_id)
        return workspace_id

    def _merge_personal_workspace_aliases(
        self, user_id: int, canonical_workspace_id: str
    ) -> None:
        """Merge legacy personal-workspace IDs without losing scoped data."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT workspace_id FROM workspaces
                   WHERE user_id = %s AND display_name = '个人工作区'
                     AND workspace_id <> %s''',
                (user_id, canonical_workspace_id),
            )
            aliases = [
                str(row["workspace_id"])
                for row in cursor.fetchall()
                if row.get("workspace_id")
            ]
            if not aliases:
                return
            placeholders = ", ".join(["%s"] * len(aliases))
            for table_name in (
                "conversations",
                "workspace_memories",
                "tasks",
                "trajectory_events",
                "background_task_runs",
            ):
                cursor.execute(
                    f'''UPDATE {table_name}
                        SET workspace_id = %s
                        WHERE user_id = %s
                          AND workspace_id IN ({placeholders})''',
                    (canonical_workspace_id, user_id, *aliases),
                )
            cursor.execute(
                f'''DELETE FROM workspaces
                    WHERE user_id = %s
                      AND workspace_id IN ({placeholders})''',
                (user_id, *aliases),
            )
        self.connection.commit()

    def create_workspace(self, user_id: int, display_name: str) -> str:
        display_name = str(display_name or "").strip()
        if not display_name or len(display_name) > 100:
            raise ValueError("工作区名称长度应为 1-100 个字符")
        workspace_id = f"ws-{uuid.uuid4().hex[:16]}"
        return self.ensure_workspace(user_id, workspace_id, display_name)

    def rename_workspace(
        self, user_id: int, workspace_id: str, display_name: str
    ) -> bool:
        """Rename a workspace owned by the current user."""
        display_name = str(display_name or "").strip()
        if not display_name or len(display_name) > 100:
            raise ValueError("工作区名称长度应为 1-100 个字符")
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT display_name FROM workspaces
                   WHERE user_id = %s AND workspace_id = %s
                   LIMIT 1 FOR UPDATE''',
                (user_id, workspace_id),
            )
            workspace = cursor.fetchone()
            if not workspace:
                self.connection.commit()
                return False
            if workspace.get("display_name") != display_name:
                cursor.execute(
                    '''UPDATE workspaces SET display_name = %s
                       WHERE user_id = %s AND workspace_id = %s''',
                    (display_name, user_id, workspace_id),
                )
        self.connection.commit()
        return True

    def list_workspaces(self, user_id: int) -> List[Dict[str, Any]]:
        self.ensure_personal_workspace(user_id)
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT workspace_id, display_name, created_at
                   FROM workspaces
                   WHERE user_id = %s
                   ORDER BY (display_name = '个人工作区') DESC,
                            created_at ASC, id ASC''',
                (user_id,),
            )
            return list(cursor.fetchall())

    def workspace_belongs_to_user(self, user_id: int, workspace_id: str) -> bool:
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT 1 FROM workspaces
                   WHERE user_id = %s AND workspace_id = %s LIMIT 1''',
                (user_id, workspace_id),
            )
            return cursor.fetchone() is not None

    def delete_workspace(self, user_id: int, workspace_id: str) -> bool:
        """Delete one custom workspace and its user-visible persisted state.

        The built-in personal workspace remains as the account's safe default.
        Physical task artifacts and append-only learning telemetry are retained;
        workspace memories, conversations and recoverable UI run state are
        removed atomically.
        """
        try:
            if workspace_id == personal_workspace_id(user_id):
                raise ValueError("系统默认的个人工作区不能删除")

            with self.connection.cursor() as cursor:
                cursor.execute(
                    '''SELECT display_name FROM workspaces
                       WHERE user_id = %s AND workspace_id = %s
                       LIMIT 1 FOR UPDATE''',
                    (user_id, workspace_id),
                )
                workspace = cursor.fetchone()
                if not workspace:
                    self.connection.commit()
                    return False

                cursor.execute(
                    '''SELECT job_id FROM background_task_runs
                       WHERE user_id = %s AND workspace_id = %s
                         AND status IN ('pending', 'running')
                       LIMIT 1''',
                    (user_id, workspace_id),
                )
                if cursor.fetchone():
                    raise ValueError("当前工作区仍有任务正在执行，完成后才能删除")

                cursor.execute(
                    '''DELETE revision
                       FROM workspace_memory_revisions AS revision
                       INNER JOIN workspace_memories AS memory
                           ON memory.id = revision.memory_id
                       WHERE memory.user_id = %s
                         AND memory.workspace_id = %s''',
                    (user_id, workspace_id),
                )
                cursor.execute(
                    '''DELETE summary
                       FROM conversation_summaries AS summary
                       INNER JOIN conversations AS conversation
                           ON conversation.id = summary.conversation_id
                       WHERE conversation.user_id = %s
                         AND conversation.workspace_id = %s''',
                    (user_id, workspace_id),
                )
                cursor.execute(
                    '''DELETE message
                       FROM messages AS message
                       INNER JOIN conversations AS conversation
                           ON conversation.id = message.conversation_id
                       WHERE conversation.user_id = %s
                         AND conversation.workspace_id = %s''',
                    (user_id, workspace_id),
                )
                cursor.execute(
                    '''DELETE indexed_file
                       FROM user_files AS indexed_file
                       INNER JOIN conversations AS conversation
                           ON conversation.session_id = indexed_file.session_id
                          AND conversation.user_id = indexed_file.user_id
                       WHERE conversation.user_id = %s
                         AND conversation.workspace_id = %s''',
                    (user_id, workspace_id),
                )
                cursor.execute(
                    '''DELETE FROM background_task_runs
                       WHERE user_id = %s AND workspace_id = %s''',
                    (user_id, workspace_id),
                )
                cursor.execute(
                    '''DELETE FROM conversations
                       WHERE user_id = %s AND workspace_id = %s''',
                    (user_id, workspace_id),
                )
                cursor.execute(
                    '''DELETE FROM workspace_memories
                       WHERE user_id = %s AND workspace_id = %s''',
                    (user_id, workspace_id),
                )
                cursor.execute(
                    '''DELETE FROM workspaces
                       WHERE user_id = %s AND workspace_id = %s''',
                    (user_id, workspace_id),
                )
                deleted = cursor.rowcount > 0
            self.connection.commit()
            return deleted
        except Exception:
            self.connection.rollback()
            raise

    def list_workspace_memories(
        self,
        user_id: int,
        workspace_id: str,
        *,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.workspace_belongs_to_user(user_id, workspace_id):
            raise ValueError("workspace does not belong to the current user")
        safe_limit = max(1, min(int(limit), 200))
        active_clause = "AND active = 1" if active_only else ""
        with self.connection.cursor() as cursor:
            cursor.execute(
                f'''SELECT id, memory_type, content, source, active,
                           created_at, updated_at
                    FROM workspace_memories
                    WHERE user_id = %s AND workspace_id = %s
                      {active_clause}
                    ORDER BY updated_at DESC, id DESC
                    LIMIT %s''',
                (user_id, workspace_id, safe_limit),
            )
            return list(cursor.fetchall())

    def get_workspace_memory(
        self, user_id: int, workspace_id: str, memory_id: int
    ) -> Dict[str, Any] | None:
        if not self.workspace_belongs_to_user(user_id, workspace_id):
            raise ValueError("workspace does not belong to the current user")
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT id, memory_type, content, source, active,
                          created_at, updated_at
                   FROM workspace_memories
                   WHERE id = %s AND user_id = %s AND workspace_id = %s
                   LIMIT 1''',
                (memory_id, user_id, workspace_id),
            )
            return cursor.fetchone()

    def add_workspace_memory(
        self,
        user_id: int,
        workspace_id: str,
        memory_type: str,
        content: str,
        *,
        source: str = "human",
    ) -> int:
        if not self.workspace_belongs_to_user(user_id, workspace_id):
            raise ValueError("workspace does not belong to the current user")
        memory_type, content = normalize_workspace_memory(memory_type, content)
        if source not in {"human", "summary", "system"}:
            raise ValueError("invalid workspace memory source")
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT INTO workspace_memories
                   (user_id, workspace_id, memory_type, content, source)
                   VALUES (%s, %s, %s, %s, %s)''',
                (user_id, workspace_id, memory_type, content, source),
            )
            memory_id = cursor.lastrowid
            cursor.execute(
                '''INSERT INTO workspace_memory_revisions
                   (memory_id, action, memory_type, content)
                   VALUES (%s, 'created', %s, %s)''',
                (memory_id, memory_type, content),
            )
            self.connection.commit()
        return memory_id

    def update_workspace_memory(
        self,
        user_id: int,
        workspace_id: str,
        memory_id: int,
        memory_type: str,
        content: str,
    ) -> bool:
        memory_type, content = normalize_workspace_memory(memory_type, content)
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT id FROM workspace_memories
                   WHERE id = %s AND user_id = %s AND workspace_id = %s
                     AND active = 1 FOR UPDATE''',
                (memory_id, user_id, workspace_id),
            )
            if cursor.fetchone() is None:
                return False
            cursor.execute(
                '''UPDATE workspace_memories
                   SET memory_type = %s, content = %s
                   WHERE id = %s''',
                (memory_type, content, memory_id),
            )
            cursor.execute(
                '''INSERT INTO workspace_memory_revisions
                   (memory_id, action, memory_type, content)
                   VALUES (%s, 'updated', %s, %s)''',
                (memory_id, memory_type, content),
            )
            self.connection.commit()
        return True

    def forget_workspace_memory(
        self, user_id: int, workspace_id: str, memory_id: int
    ) -> bool:
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT memory_type, content FROM workspace_memories
                   WHERE id = %s AND user_id = %s AND workspace_id = %s
                     AND active = 1 FOR UPDATE''',
                (memory_id, user_id, workspace_id),
            )
            record = cursor.fetchone()
            if record is None:
                return False
            cursor.execute(
                'UPDATE workspace_memories SET active = 0 WHERE id = %s',
                (memory_id,),
            )
            cursor.execute(
                '''INSERT INTO workspace_memory_revisions
                   (memory_id, action, memory_type, content)
                   VALUES (%s, 'forgotten', %s, %s)''',
                (memory_id, record['memory_type'], record['content']),
            )
            self.connection.commit()
        return True

    def get_workspace_optimization_summary(
        self,
        user_id: int,
        workspace_id: str,
        agent_version: str,
    ) -> Dict[str, Any]:
        """Return the small read-only dashboard used by the workspace UI."""
        if not self.workspace_belongs_to_user(user_id, workspace_id):
            raise ValueError("workspace does not belong to the current user")
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT data_partition, COUNT(*) AS task_count,
                          SUM(status = 'completed') AS completed_count
                   FROM tasks
                   WHERE user_id = %s AND workspace_id = %s
                     AND agent_version = %s
                   GROUP BY data_partition''',
                (user_id, workspace_id, agent_version),
            )
            partitions = {
                str(row.get('data_partition') or 'unknown'): {
                    'task_count': int(row.get('task_count') or 0),
                    'completed_count': int(row.get('completed_count') or 0),
                }
                for row in cursor.fetchall()
            }
            cursor.execute(
                '''SELECT COUNT(DISTINCT t.trace_id) AS feedback_count
                   FROM tasks t
                   JOIN task_feedback f ON f.trace_id = t.trace_id
                   WHERE t.user_id = %s AND t.workspace_id = %s
                     AND t.agent_version = %s
                     AND f.feedback_source = 'human' ''',
                (user_id, workspace_id, agent_version),
            )
            feedback = cursor.fetchone() or {}
            cursor.execute(
                '''SELECT trace_id, task_fingerprint, data_partition, status,
                          policy_id, created_at
                   FROM tasks
                   WHERE user_id = %s AND workspace_id = %s
                     AND agent_version = %s
                   ORDER BY created_at DESC, id DESC LIMIT 1''',
                (user_id, workspace_id, agent_version),
            )
            latest = cursor.fetchone()
        return {
            'partitions': partitions,
            'human_feedback_count': int(feedback.get('feedback_count') or 0),
            'latest_task': latest,
        }

    def get_user_default_session_id(
        self, user_id: int, workspace_id: str = None
    ) -> str:
        workspace_id = workspace_id or self.ensure_personal_workspace(user_id)
        if not self.workspace_belongs_to_user(user_id, workspace_id):
            raise ValueError("workspace does not belong to the current user")
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT session_id FROM conversations
                   WHERE user_id = %s AND workspace_id = %s
                   ORDER BY updated_at DESC, id DESC LIMIT 1''',
                (user_id, workspace_id)
            )
            result = cursor.fetchone()
            if result:
                return result['session_id']
            else:
                session_id = str(uuid.uuid4())
                cursor.execute(
                    '''INSERT INTO conversations
                       (user_id, workspace_id, session_id) VALUES (%s, %s, %s)''',
                    (user_id, workspace_id, session_id)
                )
                self.connection.commit()
                return session_id

    def create_workspace_session(self, user_id: int, workspace_id: str) -> str:
        """Create an empty session in a workspace and make it the latest one."""
        if not self.workspace_belongs_to_user(user_id, workspace_id):
            raise ValueError("workspace does not belong to the current user")
        session_id = str(uuid.uuid4())
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT INTO conversations
                   (user_id, workspace_id, session_id, title)
                   VALUES (%s, %s, %s, %s)''',
                (user_id, workspace_id, session_id, "新会话"),
            )
        self.connection.commit()
        return session_id

    def list_workspace_sessions(
        self, user_id: int, workspace_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return recent sessions for the selected workspace only."""
        if not self.workspace_belongs_to_user(user_id, workspace_id):
            raise ValueError("workspace does not belong to the current user")
        bounded_limit = max(1, min(int(limit), 200))
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT c.session_id, c.title, c.created_at, c.updated_at,
                          (SELECT m.content
                           FROM messages m
                           WHERE m.conversation_id = c.id AND m.role = 'user'
                           ORDER BY m.created_at ASC, m.id ASC LIMIT 1)
                              AS first_user_message
                   FROM conversations c
                   WHERE c.user_id = %s AND c.workspace_id = %s
                   ORDER BY c.updated_at DESC, c.id DESC
                   LIMIT %s''',
                (user_id, workspace_id, bounded_limit),
            )
            rows = cursor.fetchall()
        for row in rows:
            raw_title = row.get("title")
            if not raw_title or raw_title == "新会话":
                raw_title = row.get("first_user_message") or raw_title
            row["title"] = derive_conversation_title(raw_title)
            row.pop("first_user_message", None)
        return rows

    def session_belongs_to_workspace(
        self, user_id: int, workspace_id: str, session_id: str
    ) -> bool:
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT 1 FROM conversations
                   WHERE user_id = %s AND workspace_id = %s AND session_id = %s
                   LIMIT 1''',
                (user_id, workspace_id, session_id),
            )
            return cursor.fetchone() is not None

    def delete_workspace_session(
        self, user_id: int, workspace_id: str, session_id: str
    ) -> bool:
        """Delete one completed, user-visible conversation atomically.

        Task-learning trajectories are retained for workspace optimization,
        while the conversation, its messages, summaries, run snapshots and
        file index entries are removed.  Physical task artifacts are left on
        disk so deleting a history row never destroys user files implicitly.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    '''SELECT id FROM conversations
                       WHERE user_id = %s AND workspace_id = %s
                         AND session_id = %s
                       LIMIT 1 FOR UPDATE''',
                    (user_id, workspace_id, session_id),
                )
                conversation = cursor.fetchone()
                if not conversation:
                    self.connection.commit()
                    return False

                cursor.execute(
                    '''SELECT job_id FROM background_task_runs
                       WHERE user_id = %s AND workspace_id = %s
                         AND session_id = %s
                         AND status IN ('pending', 'running')
                       LIMIT 1''',
                    (user_id, workspace_id, session_id),
                )
                if cursor.fetchone():
                    raise ValueError("当前会话仍有任务正在执行，完成后才能删除")

                conversation_id = conversation['id']
                cursor.execute(
                    'DELETE FROM conversation_summaries WHERE conversation_id = %s',
                    (conversation_id,),
                )
                cursor.execute(
                    'DELETE FROM messages WHERE conversation_id = %s',
                    (conversation_id,),
                )
                cursor.execute(
                    '''DELETE FROM background_task_runs
                       WHERE user_id = %s AND workspace_id = %s
                         AND session_id = %s''',
                    (user_id, workspace_id, session_id),
                )
                cursor.execute(
                    '''DELETE FROM user_files
                       WHERE user_id = %s AND session_id = %s''',
                    (user_id, session_id),
                )
                cursor.execute(
                    '''DELETE FROM conversations
                       WHERE id = %s AND user_id = %s
                         AND workspace_id = %s AND session_id = %s''',
                    (conversation_id, user_id, workspace_id, session_id),
                )
                deleted = cursor.rowcount > 0
            self.connection.commit()
            return deleted
        except Exception:
            self.connection.rollback()
            raise

    def create_background_task_run(
        self,
        job_id: str,
        user_id: int,
        workspace_id: str,
        session_id: str,
        query_text: str,
    ) -> str:
        if not self.session_belongs_to_workspace(
            user_id, workspace_id, session_id
        ):
            raise ValueError("session does not belong to the current workspace")
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT INTO background_task_runs
                   (job_id, user_id, workspace_id, session_id, query_text)
                   VALUES (%s, %s, %s, %s, %s)''',
                (job_id, user_id, workspace_id, session_id, query_text),
            )
        self.connection.commit()
        return job_id

    def update_background_task_run(
        self,
        job_id: str,
        user_id: int,
        status: str = None,
        progress_text: str = None,
        response_text: str = None,
        error_message: str = None,
        output_files: List[str] = None,
    ) -> bool:
        allowed_statuses = {"pending", "running", "completed", "failed"}
        if status is not None and status not in allowed_statuses:
            raise ValueError("invalid background task status")
        updates = []
        params = []
        if status is not None:
            updates.append("status = %s")
            params.append(status)
            if status == "running":
                updates.append("started_at = COALESCE(started_at, NOW())")
            if status in {"completed", "failed"}:
                updates.append("completed_at = NOW()")
        if progress_text is not None:
            updates.append("progress_text = %s")
            params.append(progress_text)
        if response_text is not None:
            updates.append("response_text = %s")
            params.append(response_text)
        if error_message is not None:
            updates.append("error_message = %s")
            params.append(error_message)
        if output_files is not None:
            updates.append("output_files_json = %s")
            params.append(json.dumps(output_files, ensure_ascii=False))
        if not updates:
            return False
        params.extend([job_id, user_id])
        with self.connection.cursor() as cursor:
            cursor.execute(
                f'''UPDATE background_task_runs
                    SET {", ".join(updates)}
                    WHERE job_id = %s AND user_id = %s''',
                params,
            )
            updated = cursor.rowcount > 0
        self.connection.commit()
        return updated

    @staticmethod
    def _normalize_background_task_row(row):
        if not row:
            return row
        raw_files = row.get("output_files_json")
        if isinstance(raw_files, str):
            try:
                row["output_files"] = json.loads(raw_files)
            except (TypeError, ValueError):
                row["output_files"] = []
        elif isinstance(raw_files, list):
            row["output_files"] = raw_files
        else:
            row["output_files"] = []
        return row

    def get_background_task_run(self, user_id: int, job_id: str):
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT * FROM background_task_runs
                   WHERE user_id = %s AND job_id = %s LIMIT 1''',
                (user_id, job_id),
            )
            row = cursor.fetchone()
        return self._normalize_background_task_row(row)

    def get_active_background_task(self, user_id: int):
        """Return the user's pending/running job, if any.

        The workbench intentionally permits only one active automation per
        user.  Keeping this lookup in the persistence layer lets a newly
        opened browser page recover the same lock state as the worker.
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT * FROM background_task_runs
                   WHERE user_id = %s AND status IN ('pending', 'running')
                   ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END,
                            created_at ASC, job_id ASC
                   LIMIT 1''',
                (user_id,),
            )
            row = cursor.fetchone()
        return self._normalize_background_task_row(row)

    def get_latest_session_background_task(
        self, user_id: int, workspace_id: str, session_id: str
    ):
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT * FROM background_task_runs
                   WHERE user_id = %s AND workspace_id = %s
                     AND session_id = %s
                   ORDER BY created_at DESC, job_id DESC LIMIT 1''',
                (user_id, workspace_id, session_id),
            )
            row = cursor.fetchone()
        return self._normalize_background_task_row(row)

    def list_session_output_files(
        self, user_id: int, session_id: str, limit: int = 100
    ) -> List[str]:
        bounded_limit = max(1, min(int(limit), 200))
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT file_path FROM user_files
                   WHERE user_id = %s AND session_id = %s
                     AND file_type = 'output'
                   ORDER BY created_at DESC, id DESC LIMIT %s''',
                (user_id, session_id, bounded_limit),
            )
            return [
                str(row["file_path"])
                for row in cursor.fetchall()
                if row.get("file_path")
            ]

    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def register_user(self, username: str, password: str) -> bool:
        with self.connection.cursor() as cursor:
            cursor.execute('SELECT id FROM users WHERE username = %s', (username,))
            if cursor.fetchone():
                return False
            hashed_pwd = self.hash_password(password)
            cursor.execute('INSERT INTO users (username, password) VALUES (%s, %s)', (username, hashed_pwd))
            self.connection.commit()
            return True

    def validate_user(self, username: str, password: str) -> bool:
        hashed_pwd = self.hash_password(password)
        with self.connection.cursor() as cursor:
            cursor.execute('SELECT id FROM users WHERE username = %s AND password = %s', (username, hashed_pwd))
            return cursor.fetchone() is not None

    def find_user_id(self, username: str) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute('SELECT id FROM users WHERE username = %s', (username,))
            result = cursor.fetchone()
            if result:
                return result['id']
            return None

    def get_or_create_user_id(self, username: str) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute('SELECT id FROM users WHERE username = %s', (username,))
            result = cursor.fetchone()
            if result:
                return result['id']
            cursor.execute('INSERT INTO users (username, password) VALUES (%s, %s)', (username, ""))
            self.connection.commit()
            return cursor.lastrowid

    def get_conversation_id(
        self,
        user_id: int,
        session_id: str,
        workspace_id: str = None,
    ) -> int:
        workspace_id = (
            workspace_id
            or get_request_context().get('workspace_id')
            or self.ensure_personal_workspace(user_id)
        )
        if not self.workspace_belongs_to_user(user_id, workspace_id):
            raise ValueError("workspace does not belong to the current user")
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT id FROM conversations
                   WHERE user_id = %s AND workspace_id = %s AND session_id = %s''',
                (user_id, workspace_id, session_id)
            )
            result = cursor.fetchone()
            if result:
                return result['id']
            else:
                cursor.execute(
                    '''INSERT INTO conversations
                       (user_id, workspace_id, session_id) VALUES (%s, %s, %s)''',
                    (user_id, workspace_id, session_id)
                )
                self.connection.commit()
                return cursor.lastrowid

    def save_message(self, user_id: int, session_id: str, role: str, content: str) -> None:
        conversation_id = self.get_conversation_id(user_id, session_id)
        with self.connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)',
                (conversation_id, role, content)
            )
            if role == "user":
                cursor.execute(
                    '''UPDATE conversations
                       SET updated_at = NOW(),
                           title = CASE
                               WHEN title IS NULL OR title = '' OR title = '新会话'
                               THEN %s ELSE title END
                       WHERE id = %s''',
                    (derive_conversation_title(content), conversation_id),
                )
            else:
                cursor.execute(
                    "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
                    (conversation_id,),
                )
            self.connection.commit()

    def load_chat_history(self, user_id: int, session_id: str) -> List[Dict[str, Any]]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''
                SELECT role, content FROM messages
                JOIN conversations ON messages.conversation_id = conversations.id
                WHERE conversations.user_id = %s AND conversations.session_id = %s
                ORDER BY messages.created_at ASC
                ''',
                (user_id, session_id)
            )
            return cursor.fetchall()

    def get_conversation_messages(self, user_id: int, session_id: str):
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''
                SELECT messages.id, messages.role, messages.content, messages.created_at
                FROM messages
                         JOIN conversations ON messages.conversation_id = conversations.id
                WHERE conversations.user_id = %s
                  AND conversations.session_id = %s
                ORDER BY messages.created_at ASC, messages.id ASC
                ''',
                (user_id, session_id)
            )
            return cursor.fetchall()

    def replace_conversation_with_messages(self, user_id: int, session_id: str, new_messages: list):
        conversation_id = self.get_conversation_id(user_id, session_id)
        with self.connection.cursor() as cursor:
            cursor.execute('DELETE FROM messages WHERE conversation_id = %s', (conversation_id,))
            if new_messages:
                data = [(conversation_id, m['role'], m['content']) for m in new_messages]
                cursor.executemany(
                    'INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)',
                    data
                )
        self.connection.commit()

    def clear_user_messages(self, username: str) -> bool:
        try:
            user_id = self.find_user_id(username)
            if not user_id:
                return False

            with self.connection.cursor() as cursor:
                cursor.execute('SELECT id FROM conversations WHERE user_id = %s', (user_id,))
                conversations = cursor.fetchall()

                for conv in conversations:
                    cursor.execute('DELETE FROM messages WHERE conversation_id = %s', (conv['id'],))

                self.connection.commit()
                return True
        except Exception as e:
            print(f"清除用户消息时出错: {str(e)}")
            return False

    # ===================== 新增: 工具调用日志 =====================

    def save_tool_call_log(
        self,
        user_id: int,
        session_id: str,
        agent_name: str,
        tool_name: str,
        tool_arguments: dict,
        execution_time_ms: int,
        status: str,
        error_message: str = None,
        result_summary: str = None,
        trace_id: str = None,
        error_category: str = None,
        policy_id: str = None,
    ):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    '''INSERT INTO tool_calls_log
                       (user_id, session_id, agent_name, tool_name, tool_arguments,
                        execution_time_ms, status, error_message, result_summary,
                        trace_id, error_category, policy_id)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                    (user_id, session_id, agent_name, tool_name,
                     json.dumps(tool_arguments, ensure_ascii=False),
                     execution_time_ms, status, error_message, result_summary,
                     trace_id, error_category, policy_id or current_policy_id())
                )
                self.connection.commit()
        finally:
            pass  # caller manages connection lifecycle

    # ===================== 新增: 路由日志 =====================

    def save_agent_routing_log(
        self,
        user_id: int,
        session_id: str,
        user_query: str,
        routed_agent: str,
        confidence: float,
        reasoning: str,
        final_output_agent: str = None,
        total_latency_ms: int = None,
        trace_id: str = None,
        policy_id: str = None,
    ):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    '''INSERT INTO agent_routing_log
                       (user_id, session_id, user_query, routed_agent, confidence,
                        reasoning, final_output_agent, total_latency_ms, trace_id,
                        policy_id)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                    (user_id, session_id, user_query, routed_agent, confidence,
                     reasoning, final_output_agent, total_latency_ms, trace_id,
                     policy_id or current_policy_id())
                )
                self.connection.commit()
            return cursor.lastrowid
        finally:
            pass

    def update_routing_log(self, routing_id: int, final_output_agent: str, total_latency_ms: int):
        with self.connection.cursor() as cursor:
            cursor.execute(
                'UPDATE agent_routing_log SET final_output_agent=%s, total_latency_ms=%s WHERE id=%s',
                (final_output_agent, total_latency_ms, routing_id)
            )
            self.connection.commit()

    # ===================== 新增: 用户文件 =====================

    def save_user_file(
        self,
        user_id: int,
        session_id: str,
        file_name: str,
        file_path: str,
        file_type: str,
        file_size: int = None,
        source_tool: str = None
    ):
        if file_size is None:
            try:
                file_size = os.path.getsize(file_path)
            except Exception:
                file_size = 0

        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT INTO user_files
                   (user_id, session_id, file_name, file_path, file_size, file_type, source_tool)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)''',
                (user_id, session_id, file_name, file_path, file_size, file_type, source_tool)
            )
            self.connection.commit()

    def get_recent_session_csv_files(
        self,
        user_id: int,
        session_id: str,
        limit: int = 20,
    ) -> List[str]:
        """Return current-session CSV paths without consulting chat memory."""
        safe_limit = max(1, min(int(limit), 100))
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''SELECT file_path
                   FROM user_files
                   WHERE user_id = %s AND session_id = %s
                     AND LOWER(file_path) LIKE '%%.csv'
                   ORDER BY created_at DESC, id DESC
                   LIMIT %s''',
                (user_id, session_id, safe_limit),
            )
            return [
                str(row['file_path'])
                for row in cursor.fetchall()
                if row.get('file_path')
            ]

    # ===================== 新增: 任务记录 =====================

    def create_task(
        self,
        user_id: int,
        session_id: str,
        task_description: str,
        total_steps: int = 0,
        agent_name: str = 'ORCHESTRATOR_AGENT',
        trace_id: str = None,
        routed_agent: str = None,
        plan_json: str = None,
        policy_id: str = None,
        task_fingerprint: str = None,
        execution_fingerprint: str = None,
        workspace_id: str = None,
        agent_version: str = None,
        data_partition: str = None,
    ) -> int:
        ctx = get_request_context()
        workspace_id = (
            workspace_id
            or ctx.get('workspace_id')
            or personal_workspace_id(user_id)
        )
        agent_version = (
            agent_version
            or ctx.get('agent_version')
            or current_agent_version()
        )
        data_partition = (
            data_partition
            or ctx.get('data_partition')
            or assign_trajectory_partition(task_fingerprint or trace_id)
        )
        if data_partition not in TRAJECTORY_DATA_PARTITIONS:
            raise ValueError("invalid trajectory data_partition")
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    '''INSERT INTO tasks
                       (user_id, session_id, task_description, agent_name, status,
                        total_steps, started_at, trace_id, routed_agent, plan_json,
                        policy_id, task_fingerprint, execution_fingerprint,
                        workspace_id, agent_version, data_partition)
                       VALUES (%s, %s, %s, %s, 'running', %s, NOW(), %s, %s, %s,
                               %s, %s, %s, %s, %s, %s)''',
                    (user_id, session_id, task_description, agent_name,
                     total_steps, trace_id, routed_agent, plan_json,
                     policy_id or current_policy_id(), task_fingerprint,
                     execution_fingerprint, workspace_id, agent_version,
                     data_partition)
                )
                self.connection.commit()
                return cursor.lastrowid
        finally:
            pass

    def update_task(
        self,
        task_id: int,
        completed_steps: int = None,
        status: str = None,
        result_summary: str = None,
        error_message: str = None,
        error_category: str = None,
        plan_json: str = None,
        total_steps: int = None,
        routed_agent: str = None,
        task_fingerprint: str = None,
        execution_fingerprint: str = None,
        outcome_status: str = None,
        completion_gate_status: str = None,
        outcome_summary: dict = None,
    ):
        if status == 'completed' and completion_gate_status != 'pass':
            raise ValueError(
                "completed status requires completion_gate_status='pass'"
            )
        try:
            with self.connection.cursor() as cursor:
                updates = []
                params = []
                if completed_steps is not None:
                    updates.append('completed_steps = %s')
                    params.append(completed_steps)
                if status is not None:
                    updates.append('status = %s')
                    params.append(status)
                    if status in ('completed', 'failed'):
                        updates.append('completed_at = NOW()')
                if result_summary is not None:
                    updates.append('result_summary = %s')
                    params.append(result_summary)
                if error_message is not None:
                    updates.append('error_message = %s')
                    params.append(error_message)
                if error_category is not None:
                    updates.append('error_category = %s')
                    params.append(error_category)
                if plan_json is not None:
                    updates.append('plan_json = %s')
                    params.append(plan_json)
                if total_steps is not None:
                    updates.append('total_steps = %s')
                    params.append(total_steps)
                if routed_agent is not None:
                    updates.append('routed_agent = %s')
                    params.append(routed_agent)
                if task_fingerprint is not None:
                    updates.append('task_fingerprint = %s')
                    params.append(task_fingerprint)
                if execution_fingerprint is not None:
                    updates.append('execution_fingerprint = %s')
                    params.append(execution_fingerprint)
                if outcome_status is not None:
                    updates.append('outcome_status = %s')
                    params.append(outcome_status)
                if completion_gate_status is not None:
                    updates.append('completion_gate_status = %s')
                    params.append(completion_gate_status)
                if outcome_summary is not None:
                    updates.append('outcome_summary_json = %s')
                    params.append(
                        json.dumps(
                            sanitize_for_audit(outcome_summary),
                            ensure_ascii=False,
                        )
                    )

                if updates:
                    params.append(task_id)
                    cursor.execute(
                        f'UPDATE tasks SET {", ".join(updates)} WHERE id = %s',
                        params
                    )
                    self.connection.commit()
        finally:
            pass

    # ===================== 新增: 会话摘要 =====================

    def save_conversation_summary(
        self,
        user_id: int,
        session_id: str,
        summary_content: str,
        summarized_message_count: int
    ):
        conversation_id = self.get_conversation_id(user_id, session_id)
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT INTO conversation_summaries
                   (conversation_id, summary_content, summarized_message_count)
                   VALUES (%s, %s, %s)''',
                (conversation_id, summary_content, summarized_message_count)
            )
            self.connection.commit()

    # ===================== 新增: Agent 执行日志 =====================

    def save_agent_execution_log(
        self,
        user_id: int,
        session_id: str,
        agent_name: str,
        trace_id: str = None,
        total_latency_ms: int = None,
        llm_latency_ms: int = None,
        tool_count: int = 0,
        success: bool = True,
        error_category: str = None,
        error_message: str = None,
        policy_id: str = None,
        model_name: str = None,
        prompt_hash: str = None,
        input_tokens: int = None,
        output_tokens: int = None,
        estimated_cost: float = None,
    ) -> int:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    '''INSERT INTO agent_execution_log
                       (trace_id, user_id, session_id, agent_name,
                        total_latency_ms, llm_latency_ms, tool_count,
                        success, error_category, error_message, policy_id,
                        model_name, prompt_hash, input_tokens, output_tokens,
                        estimated_cost)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s)''',
                    (trace_id, user_id, session_id, agent_name,
                     total_latency_ms, llm_latency_ms, tool_count,
                      1 if success else 0, error_category, error_message,
                      policy_id or current_policy_id(), model_name, prompt_hash,
                      input_tokens, output_tokens, estimated_cost)
                )
                self.connection.commit()
                return cursor.lastrowid
        finally:
            pass

    # ===================== 新增: 任务步骤明细 =====================

    def save_task_step(
        self,
        task_id: int,
        step_index: int,
        agent_name: str,
        task_description: str = None,
        status: str = 'success',
        trace_id: str = None,
        error_category: str = None,
        error_message: str = None,
        output_summary: str = None,
    ) -> int:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    '''INSERT INTO task_steps
                       (task_id, trace_id, step_index, agent_name,
                        task_description, status, error_category,
                        error_message, output_summary)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                    (task_id, trace_id, step_index, agent_name,
                     task_description, status, error_category,
                     error_message, output_summary)
                )
                self.connection.commit()
                return cursor.lastrowid
        finally:
            pass

    # ===================== 新增: 实验结果（upsert） =====================

    def save_experiment_result(self, row: dict) -> int:
        """插入或更新一条实验结果。按 (output_dir, suffix) 去重。"""
        try:
            with self.connection.cursor() as cursor:
                columns = [
                    'trace_id', 'user_id', 'session_id', 'output_dir', 'suffix',
                    'feature_type', 'classification_method',
                    'train_path', 'test_path', 'train_sha256', 'test_sha256',
                    'train_rows', 'train_unique_patients',
                    'test_rows', 'test_unique_patients',
                    'NumPatients', 'AUC', 'PR_AUC', 'Accuracy', 'Precision',
                    'Sensitivity', 'Specificity', 'F1',
                    'TN', 'FP', 'FN', 'TP',
                    'selected_feature_count', 'final_c', 'random_seed',
                    'python_version', 'numpy_version', 'pandas_version',
                    'sklearn_version', 'manifest_sha256',
                ]
                placeholders = ', '.join(['%s'] * len(columns))
                col_names = ', '.join(f'`{c}`' for c in columns)
                values = [row.get(c) for c in columns]

                # ON DUPLICATE KEY UPDATE: 刷新所有度量值列
                metric_columns = [
                    'AUC', 'PR_AUC', 'Accuracy', 'Precision',
                    'Sensitivity', 'Specificity', 'F1',
                    'TN', 'FP', 'FN', 'TP', 'NumPatients',
                    'selected_feature_count', 'final_c',
                    'updated_at',
                ]
                update_clause = ', '.join(
                    f'`{c}` = VALUES(`{c}`)' for c in metric_columns
                )
                update_clause += ', updated_at = NOW()'

                cursor.execute(
                    f'''INSERT INTO experiment_results
                        ({col_names}) VALUES ({placeholders})
                        ON DUPLICATE KEY UPDATE {update_clause}''',
                    values
                )
                self.connection.commit()
                return cursor.lastrowid
        finally:
            pass

    # ===================== Unified trajectory and optimization records =====================

    def save_trajectory_event(self, event: dict) -> str:
        """Append one normalized, sanitized trajectory event."""
        required = {
            'event_id', 'schema_version', 'trace_id', 'event_type',
            'actor_type', 'actor_name', 'status', 'payload', 'occurred_at',
        }
        missing = required.difference(event)
        if missing:
            raise ValueError(f"trajectory event missing fields: {sorted(missing)}")

        payload = sanitize_for_audit(event.get('payload', {}))
        ctx = get_request_context()
        user_id = ctx.get('user_id') or event.get('user_id')
        workspace_id = ctx.get('workspace_id') or event.get('workspace_id')
        agent_version = (
            ctx.get('agent_version')
            or event.get('agent_version')
            or current_agent_version()
        )
        data_partition = ctx.get('data_partition') or event.get('data_partition')
        if data_partition is not None and data_partition not in TRAJECTORY_DATA_PARTITIONS:
            raise ValueError("invalid trajectory data_partition")
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT IGNORE INTO trajectory_events
                   (event_id, schema_version, trace_id, task_id, step_id,
                    parent_event_id, event_type, actor_type, actor_name,
                    policy_id, user_id, workspace_id, agent_version,
                    data_partition, status, duration_ms, payload_json, occurred_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s)''',
                (
                    event['event_id'], event['schema_version'], event['trace_id'],
                    event.get('task_id'), event.get('step_id'),
                    event.get('parent_event_id'), event['event_type'],
                    event['actor_type'], event['actor_name'], event.get('policy_id'),
                    user_id, workspace_id, agent_version, data_partition,
                    event['status'], event.get('duration_ms'),
                    json.dumps(payload, ensure_ascii=False), event['occurred_at'],
                ),
            )
            self.connection.commit()
        return event['event_id']

    def save_pipeline_trial(self, record: dict) -> str:
        """Idempotently append one typed candidate-level trial record."""
        required = {
            'trial_id', 'schema_version', 'trace_id', 'candidate_id',
            'feature_type', 'classification_method', 'model_recipe',
            'candidate_config_sha256', 'status', 'validation_scope',
            'test_data_used', 'recommended_params', 'metrics',
            'metric_intervals', 'details',
        }
        missing = required.difference(record)
        if missing:
            raise ValueError(f"pipeline trial missing fields: {sorted(missing)}")
        if record['status'] not in ('success', 'failed'):
            raise ValueError("pipeline trial status must be success or failed")
        if record['status'] == 'success' and record.get('test_data_used'):
            raise ValueError(
                "a successful pipeline trial cannot contain test evidence"
            )
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT IGNORE INTO pipeline_trials
                   (trial_id, schema_version, trace_id, task_id,
                    task_fingerprint, dataset_hash, candidate_id, feature_type,
                    classification_method, model_recipe,
                    candidate_config_sha256, policy_id, status,
                    validation_scope, test_data_used, evaluator_version,
                    train_sha256, patient_count, outer_folds,
                    validation_score, auc, pr_auc, f1, auc_ci_lower,
                    auc_ci_upper, selection_stability, duration_ms,
                    recommended_params_json, metrics_json,
                    metric_intervals_json, details_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s)''',
                (
                    record['trial_id'], record['schema_version'],
                    record['trace_id'], record.get('task_id'),
                    record.get('task_fingerprint'), record.get('dataset_hash'),
                    record['candidate_id'], record['feature_type'],
                    record['classification_method'], record['model_recipe'],
                    record['candidate_config_sha256'], record.get('policy_id'),
                    record['status'], record['validation_scope'],
                    int(bool(record.get('test_data_used'))),
                    record.get('evaluator_version'), record.get('train_sha256'),
                    record.get('patient_count'), record.get('outer_folds'),
                    record.get('validation_score'), record.get('auc'),
                    record.get('pr_auc'), record.get('f1'),
                    record.get('auc_ci_lower'), record.get('auc_ci_upper'),
                    record.get('selection_stability'), record.get('duration_ms'),
                    json.dumps(
                        sanitize_for_audit(record.get('recommended_params', {})),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        sanitize_for_audit(record.get('metrics', {})),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        sanitize_for_audit(record.get('metric_intervals', {})),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        sanitize_for_audit(record.get('details', {})),
                        ensure_ascii=False,
                    ),
                ),
            )
            self.connection.commit()
        return record['trial_id']

    def register_artifact(self, record: dict) -> str:
        """Upsert a content-addressed artifact without overwriting its identity."""
        required = {'artifact_id', 'artifact_type', 'uri', 'sha256', 'created_by'}
        missing = required.difference(record)
        if missing:
            raise ValueError(f"artifact record missing fields: {sorted(missing)}")

        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT INTO artifact_registry
                   (artifact_id, trace_id, task_id, artifact_type, uri, sha256,
                    created_by, source_artifact_ids, metadata_json,
                    validation_status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                     trace_id = COALESCE(VALUES(trace_id), trace_id),
                     task_id = COALESCE(VALUES(task_id), task_id),
                     metadata_json = VALUES(metadata_json),
                     validation_status = VALUES(validation_status)''',
                (
                    record['artifact_id'], record.get('trace_id'),
                    record.get('task_id'), record['artifact_type'], record['uri'],
                    record['sha256'], record['created_by'],
                    json.dumps(record.get('source_artifact_ids', []), ensure_ascii=False),
                    json.dumps(sanitize_for_audit(record.get('metadata', {})), ensure_ascii=False),
                    record.get('validation_status', 'unverified'),
                ),
            )
            self.connection.commit()
        return record['artifact_id']

    def save_evaluation_result(self, record: dict) -> str:
        """Append a versioned metric or hard-gate evaluation."""
        required = {
            'evaluation_id', 'task_type', 'split_type', 'evaluator_name',
            'evaluator_version', 'record_type', 'metric_name',
        }
        missing = required.difference(record)
        if missing:
            raise ValueError(f"evaluation record missing fields: {sorted(missing)}")
        if record['record_type'] not in ('metric', 'gate'):
            raise ValueError("record_type must be 'metric' or 'gate'")

        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT INTO evaluation_results
                   (evaluation_id, trace_id, task_id, artifact_id, policy_id,
                    task_type, method_name, split_type, dataset_hash, evaluator_name,
                    evaluator_version, record_type, metric_name, metric_value,
                    gate_status, details_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                     metric_value = VALUES(metric_value),
                     gate_status = VALUES(gate_status),
                     details_json = VALUES(details_json)''',
                (
                    record['evaluation_id'], record.get('trace_id'),
                    record.get('task_id'), record.get('artifact_id'),
                    record.get('policy_id'), record['task_type'],
                    record.get('method_name', 'unknown'), record['split_type'],
                    record.get('dataset_hash'),
                    record['evaluator_name'], record['evaluator_version'],
                    record['record_type'], record['metric_name'],
                    record.get('metric_value'), record.get('gate_status'),
                    json.dumps(sanitize_for_audit(record.get('details', {})), ensure_ascii=False),
                ),
            )
            self.connection.commit()
        return record['evaluation_id']

    def register_policy_version(self, record: dict) -> str:
        """Register a policy candidate. Activation still requires explicit approval."""
        required = {
            'policy_id', 'policy_type', 'version', 'status', 'config',
            'config_sha256', 'created_by',
        }
        missing = required.difference(record)
        if missing:
            raise ValueError(f"policy record missing fields: {sorted(missing)}")
        record = dict(record)
        if record.get('policy_type') == 'multi_agent_shadow_candidate':
            from .self_optimization_control import (
                load_self_optimization_constitution,
                validate_candidate_policy_record,
            )

            record = validate_candidate_policy_record(
                record,
                load_self_optimization_constitution(),
            )
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT INTO policy_versions
                   (policy_id, policy_type, version, status, config_json,
                    config_sha256, parent_policy_id, created_by)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                      policy_id = policy_id''',
                (
                    record['policy_id'], record['policy_type'], record['version'],
                    record['status'],
                    json.dumps(sanitize_for_audit(record['config']), ensure_ascii=False),
                    record['config_sha256'], record.get('parent_policy_id'),
                    record['created_by'],
                ),
            )
            self.connection.commit()
        return record['policy_id']

    def save_routing_feedback(
        self,
        *,
        trace_id: str,
        expected_agent: str,
        observed_agent: str = None,
        label_source: str,
        routing_log_id: int = None,
        reviewer_id: str = None,
        notes: str = None,
    ) -> int:
        """Store an independent routing reference label for optimization."""
        if label_source not in ('human', 'rule', 'adjudicated'):
            raise ValueError("label_source must be human, rule, or adjudicated")
        if not trace_id or not expected_agent:
            raise ValueError("trace_id and expected_agent are required")

        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT INTO routing_feedback
                   (routing_log_id, trace_id, expected_agent, observed_agent,
                    label_source, reviewer_id, notes)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                     expected_agent = IF(
                       FIELD(label_source, 'rule', 'human', 'adjudicated') >
                       FIELD(VALUES(label_source), 'rule', 'human', 'adjudicated'),
                       expected_agent, VALUES(expected_agent)),
                     observed_agent = IF(
                       FIELD(label_source, 'rule', 'human', 'adjudicated') >
                       FIELD(VALUES(label_source), 'rule', 'human', 'adjudicated'),
                       observed_agent, VALUES(observed_agent)),
                     reviewer_id = IF(
                       FIELD(label_source, 'rule', 'human', 'adjudicated') >
                       FIELD(VALUES(label_source), 'rule', 'human', 'adjudicated'),
                       reviewer_id, VALUES(reviewer_id)),
                     notes = IF(
                       FIELD(label_source, 'rule', 'human', 'adjudicated') >
                       FIELD(VALUES(label_source), 'rule', 'human', 'adjudicated'),
                       notes, VALUES(notes)),
                     label_source = IF(
                       FIELD(label_source, 'rule', 'human', 'adjudicated') >
                       FIELD(VALUES(label_source), 'rule', 'human', 'adjudicated'),
                       label_source, VALUES(label_source))''',
                (
                    routing_log_id, trace_id, expected_agent, observed_agent,
                    label_source, reviewer_id,
                    str(sanitize_for_audit(notes)) if notes is not None else None,
                ),
            )
            self.connection.commit()
            return cursor.lastrowid

    def save_task_feedback(
        self,
        *,
        trace_id: str,
        outcome_label: str,
        feedback_source: str,
        task_id: int = None,
        accepted: bool = None,
        correction_required: bool = None,
        confidence: float = None,
        expected_agent: str = None,
        expected_pipeline_id: str = None,
        error_types: list = None,
        reviewer_id: str = None,
        notes: str = None,
    ) -> int:
        """Store task outcome feedback without treating inference as human truth."""
        if feedback_source not in ('human', 'rule', 'inferred', 'adjudicated'):
            raise ValueError("invalid feedback_source")
        if outcome_label not in ('correct', 'partial', 'incorrect', 'unknown'):
            raise ValueError("invalid outcome_label")
        if not trace_id:
            raise ValueError("trace_id is required")
        if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        with self.connection.cursor() as cursor:
            cursor.execute(
                '''INSERT INTO task_feedback
                   (trace_id, task_id, feedback_source, outcome_label,
                    accepted, correction_required, confidence, expected_agent,
                    expected_pipeline_id, error_types, reviewer_id, notes)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                     task_id = COALESCE(VALUES(task_id), task_id),
                     outcome_label = VALUES(outcome_label),
                     accepted = VALUES(accepted),
                     correction_required = VALUES(correction_required),
                     confidence = VALUES(confidence),
                     expected_agent = VALUES(expected_agent),
                     expected_pipeline_id = VALUES(expected_pipeline_id),
                     error_types = VALUES(error_types),
                     reviewer_id = VALUES(reviewer_id),
                     notes = VALUES(notes)''',
                (
                    trace_id, task_id, feedback_source, outcome_label,
                    None if accepted is None else int(bool(accepted)),
                    (
                        None if correction_required is None
                        else int(bool(correction_required))
                    ),
                    confidence, expected_agent, expected_pipeline_id,
                    json.dumps(
                        sanitize_for_audit(error_types or []),
                        ensure_ascii=False,
                    ),
                    reviewer_id,
                    str(sanitize_for_audit(notes)) if notes is not None else None,
                ),
            )
            self.connection.commit()
            return cursor.lastrowid

    def close(self):
        """关闭数据库连接"""
        self.connection.close()
