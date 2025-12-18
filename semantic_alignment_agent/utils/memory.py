from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class MemoryRecord:
    """单条记忆记录（用于 ReAct 流程的步骤跟踪）"""

    timestamp: str
    element_guid: str
    step_index: int
    thought: Optional[str] = None
    action: Optional[Dict[str, Any]] = None
    observation: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None


class AgentMemory:
    """Agent 内存管理：短期工作记忆 + 长期会话记忆持久化"""

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path or "logs/agent_memory.json"
        self._episode: List[MemoryRecord] = []
        self._long_term: Dict[str, List[Dict[str, Any]]] = {}

        # 尝试加载历史持久化内存
        try:
            path = Path(self.persist_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._long_term = data.get("long_term", {})
        except Exception:
            # 安静失败，保持空内存
            self._long_term = {}

    def start_episode(self):
        """清空短期工作记忆，开始新的 episode"""
        self._episode = []

    def add_plan(self, element_guid: str, step_index: int, thought: str, action: Dict[str, Any]):
        record = MemoryRecord(
            timestamp=datetime.now().isoformat(),
            element_guid=element_guid,
            step_index=step_index,
            thought=thought,
            action=action,
        )
        self._episode.append(record)

    def add_observation(self, element_guid: str, step_index: int, observation: Dict[str, Any], summary: Optional[str] = None):
        # 找到对应步骤并补充 observation
        for rec in self._episode:
            if rec.element_guid == element_guid and rec.step_index == step_index:
                rec.observation = observation
                rec.summary = summary
                break
        else:
            # 若未找到，则追加新的记录
            self._episode.append(
                MemoryRecord(
                    timestamp=datetime.now().isoformat(),
                    element_guid=element_guid,
                    step_index=step_index,
                    observation=observation,
                    summary=summary,
                )
            )

    def get_episode(self) -> List[Dict[str, Any]]:
        return [self._to_dict(rec) for rec in self._episode]

    def summarize_episode(self) -> Dict[str, Any]:
        return {
            "steps": len(self._episode),
            "records": [self._to_dict(rec) for rec in self._episode],
        }

    def commit_long_term(self, element_guid: str):
        """将当前 episode 的记录持久化到长期记忆并保存到文件"""
        episode_dict = self.get_episode()
        if element_guid not in self._long_term:
            self._long_term[element_guid] = []
        self._long_term[element_guid].append({
            "timestamp": datetime.now().isoformat(),
            "episode": episode_dict,
        })
        self._persist()

    def get_long_term(self, element_guid: str) -> List[Dict[str, Any]]:
        return self._long_term.get(element_guid, [])

    def _persist(self):
        try:
            path = Path(self.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"long_term": self._long_term}, f, ensure_ascii=False, indent=2)
        except Exception:
            # 持久化失败不影响主流程
            pass

    @staticmethod
    def _to_dict(rec: MemoryRecord) -> Dict[str, Any]:
        return {
            "timestamp": rec.timestamp,
            "element_guid": rec.element_guid,
            "step_index": rec.step_index,
            "thought": rec.thought,
            "action": rec.action,
            "observation": rec.observation,
            "summary": rec.summary,
        }