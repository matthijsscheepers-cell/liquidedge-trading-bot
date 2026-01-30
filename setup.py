"""Setup configuration for LIQUIDEDGE trading bot."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="liquidedge",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A hybrid trading bot for automated forex trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/liquidedge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "python-dotenv>=1.0.0",
        "ta>=0.11.0",
        "oandapyV20>=0.7.2",
        "backtrader>=1.9.76.123",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.75.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "liquidedge-backtest=scripts.run_backtest:main",
            "liquidedge-live=scripts.run_live:main",
            "liquidedge-fetch=scripts.fetch_data:main",
        ],
    },
    package_data={
        "liquidedge": ["config/*.json", "config/*.yaml"],
    },
    include_package_data=True,
    zip_safe=False,
)
