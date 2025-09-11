"""File utilities for IFC Semantic Agent."""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union


class FileUtils:
    """File operation utilities."""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """Ensure directory exists.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Read JSON file.
        
        Args:
            file_path: JSON file path
            
        Returns:
            JSON data as dictionary
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
        """Write data to JSON file.
        
        Args:
            data: Data to write
            file_path: Output file path
            indent: JSON indentation
        """
        file_path = Path(file_path)
        FileUtils.ensure_dir(file_path.parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def read_text(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """Read text file.
        
        Args:
            file_path: Text file path
            encoding: File encoding
            
        Returns:
            File content as string
        """
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def write_text(content: str, file_path: Union[str, Path], encoding: str = 'utf-8') -> None:
        """Write text to file.
        
        Args:
            content: Text content
            file_path: Output file path
            encoding: File encoding
        """
        file_path = Path(file_path)
        FileUtils.ensure_dir(file_path.parent)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def read_pickle(file_path: Union[str, Path]) -> Any:
        """Read pickle file.
        
        Args:
            file_path: Pickle file path
            
        Returns:
            Unpickled object
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def write_pickle(obj: Any, file_path: Union[str, Path]) -> None:
        """Write object to pickle file.
        
        Args:
            obj: Object to pickle
            file_path: Output file path
        """
        file_path = Path(file_path)
        FileUtils.ensure_dir(file_path.parent)
        
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def list_files(directory: Union[str, Path], pattern: str = '*', recursive: bool = False) -> List[Path]:
        """List files in directory.
        
        Args:
            directory: Directory path
            pattern: File pattern (glob)
            recursive: Whether to search recursively
            
        Returns:
            List of file paths
        """
        directory = Path(directory)
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """Get file size in bytes.
        
        Args:
            file_path: File path
            
        Returns:
            File size in bytes
        """
        return os.path.getsize(file_path)
    
    @staticmethod
    def file_exists(file_path: Union[str, Path]) -> bool:
        """Check if file exists.
        
        Args:
            file_path: File path
            
        Returns:
            True if file exists
        """
        return Path(file_path).exists()
    
    @staticmethod
    def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
        """Copy file.
        
        Args:
            src: Source file path
            dst: Destination file path
        """
        import shutil
        
        dst = Path(dst)
        FileUtils.ensure_dir(dst.parent)
        shutil.copy2(src, dst)
    
    @staticmethod
    def move_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
        """Move file.
        
        Args:
            src: Source file path
            dst: Destination file path
        """
        import shutil
        
        dst = Path(dst)
        FileUtils.ensure_dir(dst.parent)
        shutil.move(src, dst)
    
    @staticmethod
    def delete_file(file_path: Union[str, Path]) -> None:
        """Delete file.
        
        Args:
            file_path: File path to delete
        """
        path = Path(file_path)
        if path.exists():
            path.unlink()