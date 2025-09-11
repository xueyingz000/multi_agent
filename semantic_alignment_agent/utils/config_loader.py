import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[Dict[str, Any]] = None
    
    def _find_config_file(self) -> str:
        """查找配置文件"""
        # 查找顺序：当前目录 -> 项目根目录
        possible_paths = [
            "config.yaml",
            "../config.yaml",
            Path(__file__).parent.parent / "config.yaml"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(Path(path).resolve())
        
        raise FileNotFoundError("Configuration file not found")
    
    def load(self) -> Dict[str, Any]:
        """加载配置"""
        if self._config is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        config = self.load()
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        llm_config = self.get('llm', {})
        
        # 从环境变量覆盖配置
        if os.getenv('OPENAI_MODEL'):
            llm_config['model'] = os.getenv('OPENAI_MODEL')
        
        return llm_config
    
    def get_ifc_config(self) -> Dict[str, Any]:
        """获取IFC处理配置"""
        return self.get('ifc_processing', {})
    
    def get_semantic_config(self) -> Dict[str, Any]:
        """获取语义对齐配置"""
        return self.get('semantic_alignment', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.get('logging', {})


# 全局配置实例
config = ConfigLoader()