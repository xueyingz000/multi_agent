import os
import sys
from pathlib import Path
from loguru import logger
from typing import Optional

from .config_loader import config


class Logger:
    """日志管理器"""
    
    _initialized = False
    
    @classmethod
    def setup(cls, config_override: Optional[dict] = None) -> None:
        """设置日志配置"""
        if cls._initialized:
            return
        
        # 获取日志配置
        log_config = config_override or config.get_logging_config()
        
        # 移除默认处理器
        logger.remove()
        
        # 控制台输出
        logger.add(
            sys.stderr,
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', 
                "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"),
            colorize=True
        )
        
        # 文件输出
        log_file = log_config.get('file')
        if log_file:
            # 确保日志目录存在
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                level=log_config.get('level', 'INFO'),
                format=log_config.get('format',
                    "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"),
                rotation="10 MB",
                retention="7 days",
                compression="zip"
            )
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str = None):
        """获取日志器"""
        if not cls._initialized:
            cls.setup()
        
        if name:
            return logger.bind(name=name)
        return logger


# 初始化日志
Logger.setup()

# 导出常用的日志函数
get_logger = Logger.get_logger
log = Logger.get_logger("semantic_alignment")