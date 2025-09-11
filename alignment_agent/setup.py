#!/usr/bin/env python3
"""Setup script for IFC Semantic Agent."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Development requirements
dev_requirements = [
    'pytest>=7.0.0',
    'pytest-cov>=4.0.0',
    'pytest-asyncio>=0.21.0',
    'black>=23.0.0',
    'flake8>=6.0.0',
    'mypy>=1.0.0',
    'pre-commit>=3.0.0',
    'sphinx>=6.0.0',
    'sphinx-rtd-theme>=1.2.0',
    'jupyter>=1.0.0',
    'notebook>=6.5.0'
]

# Documentation requirements
docs_requirements = [
    'sphinx>=6.0.0',
    'sphinx-rtd-theme>=1.2.0',
    'myst-parser>=1.0.0',
    'sphinx-autodoc-typehints>=1.20.0'
]

# Testing requirements
test_requirements = [
    'pytest>=7.0.0',
    'pytest-cov>=4.0.0',
    'pytest-asyncio>=0.21.0',
    'pytest-mock>=3.10.0',
    'coverage>=7.0.0'
]

setup(
    name="ifc-semantic-agent",
    version="1.0.0",
    author="IFC Semantic Agent Team",
    author_email="contact@ifc-semantic-agent.com",
    description="Intelligent IFC-Regulatory Semantic Alignment System using ReAct Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ifc-semantic-agent/ifc-semantic-agent",
    project_urls={
        "Bug Tracker": "https://github.com/ifc-semantic-agent/ifc-semantic-agent/issues",
        "Documentation": "https://ifc-semantic-agent.readthedocs.io/",
        "Source Code": "https://github.com/ifc-semantic-agent/ifc-semantic-agent",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": docs_requirements,
        "test": test_requirements,
        "all": dev_requirements + docs_requirements + test_requirements,
    },
    entry_points={
        "console_scripts": [
            "ifc-agent=run_agent:main",
            "ifc-semantic-agent=run_agent:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml", "*.yml", "*.json"],
        "data": ["*.json", "*.yaml", "*.txt"],
        "models": ["*.pkl", "*.joblib", "*.bin"],
    },
    zip_safe=False,
    keywords=[
        "IFC", "BIM", "semantic alignment", "regulatory compliance", 
        "knowledge graph", "RAG", "ReAct", "NLP", "AI", "construction",
        "building codes", "semantic extraction", "multi-modal"
    ],
    platforms=["any"],
    license="MIT",
)