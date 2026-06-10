from setuptools import setup, find_packages

setup(
    name="meeting-intelligence",
    version="0.1.0",
    description="Real-Time Multilingual Meeting Intelligence System",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        # Core requirements managed in requirements.txt
    ],
    entry_points={
        "console_scripts": [
            "mi-download-models=scripts.download_models:main",
            "mi-benchmark=benchmarks.benchmark_runner:main",
        ],
    },
    python_requires=">=3.10",
)
