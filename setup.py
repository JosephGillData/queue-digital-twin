"""Setup script for queue-digital-twin."""

from setuptools import setup, find_packages

setup(
    name="queue-digital-twin",
    version="0.1.0",
    description="A SimPy-based digital twin that learns from a physical queue system",
    author="Queue Digital Twin",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "run-simulation=scripts.run_simulation:main",
        ],
    },
)
