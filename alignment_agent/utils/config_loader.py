"""Configuration loader for IFC Semantic Agent."""

import os
import yaml
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv


class ConfigLoader:
    """Configuration loader with environment variable support."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. Defaults to config/config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = None
        
        # Load environment variables
        load_dotenv()
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Replace environment variables
        config_content = self._replace_env_vars(config_content)
        
        self.config = yaml.safe_load(config_content)
    
    def _replace_env_vars(self, content: str) -> str:
        """Replace environment variables in configuration content.
        
        Args:
            content: Configuration file content
            
        Returns:
            Content with environment variables replaced
        """
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        return re.sub(r'\$\{([^}]+)\}', replace_var, content)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'llm.model_name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section as dictionary
        """
        return self.get(section, {})
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: str = None) -> None:
        """Save configuration to file.
        
        Args:
            output_path: Output file path. Defaults to original config path
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    @property
    def all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return self.config.copy()


# Global configuration instance
_config_instance = None


def get_config() -> ConfigLoader:
    """Get global configuration instance.
    
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance