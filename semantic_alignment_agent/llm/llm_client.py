"""LLM Client Module for Semantic Alignment Agent

This module provides a unified interface for interacting with Large Language Models,
specifically designed for semantic alignment tasks in building information modeling.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

try:
    import openai
except ImportError:
    openai = None

from ..utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class LLMClient:
    """Unified LLM client for semantic alignment tasks."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize LLM client with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path).get_config()
        self.llm_config = self.config.get('llm', {})
        
        # Initialize OpenAI client if available
        self.openai_client = None
        if openai and self.llm_config.get('provider') == 'openai':
            self._init_openai_client()
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        api_key = self.llm_config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OpenAI API key not found. LLM features will be disabled.")
            return
        
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def is_available(self) -> bool:
        """Check if LLM client is available and configured."""
        return self.openai_client is not None
    
    def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> Optional[str]:
        """Generate completion using the configured LLM.
        
        Args:
            prompt: Input prompt for the LLM
            model: Model name (defaults to config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_format: Expected response format
            
        Returns:
            Generated completion text or None if failed
        """
        if not self.is_available():
            logger.warning("LLM client not available, returning None")
            return None
        
        model = model or self.llm_config.get('model', 'gpt-3.5-turbo')
        max_tokens = max_tokens or self.llm_config.get('max_tokens', 2000)
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert in building information modeling and semantic analysis."},
                {"role": "user", "content": prompt}
            ]
            
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = self.openai_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate completion: {e}")
            return None
    
    def generate_json_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1
    ) -> Optional[Dict]:
        """Generate JSON-formatted completion.
        
        Args:
            prompt: Input prompt for the LLM
            model: Model name (defaults to config)
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON response or None if failed
        """
        response_format = {"type": "json_object"}
        
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nPlease provide your response in valid JSON format."
        
        completion = self.generate_completion(
            json_prompt,
            model=model,
            temperature=temperature,
            response_format=response_format
        )
        
        if not completion:
            return None
        
        try:
            return json.loads(completion)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None
    
    def analyze_with_confidence(
        self,
        prompt: str,
        context: Dict[str, Any],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze input with confidence scoring.
        
        Args:
            prompt: Analysis prompt
            context: Context information
            model: Model name
            
        Returns:
            Analysis result with confidence score
        """
        # Enhance prompt with confidence scoring instruction
        enhanced_prompt = f"""{prompt}
        
Context Information:
{json.dumps(context, indent=2)}

Please provide your analysis in JSON format with the following structure:
{{
    "analysis": "your detailed analysis",
    "conclusion": "your final conclusion",
    "confidence_score": 0.85,
    "confidence_factors": [
        "factor 1 that increases/decreases confidence",
        "factor 2 that increases/decreases confidence"
    ],
    "uncertainty_areas": [
        "area 1 with uncertainty",
        "area 2 with uncertainty"
    ],
    "additional_context_needed": [
        "what additional information would help"
    ]
}}

Confidence score should be between 0.0 and 1.0, where:
- 0.9-1.0: Very high confidence
- 0.7-0.9: High confidence  
- 0.5-0.7: Medium confidence
- 0.3-0.5: Low confidence
- 0.0-0.3: Very low confidence
"""
        
        result = self.generate_json_completion(enhanced_prompt, model=model)
        
        if not result:
            return {
                "analysis": "LLM analysis failed",
                "conclusion": "Unable to analyze",
                "confidence_score": 0.0,
                "confidence_factors": ["LLM unavailable"],
                "uncertainty_areas": ["Complete analysis failure"],
                "additional_context_needed": ["Working LLM connection"]
            }
        
        # Ensure required fields exist
        result.setdefault("confidence_score", 0.5)
        result.setdefault("confidence_factors", [])
        result.setdefault("uncertainty_areas", [])
        result.setdefault("additional_context_needed", [])
        
        return result