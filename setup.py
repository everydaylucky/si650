from setuptools import setup, find_packages

setup(
    name="final_test",
    version="0.1.0",
    description="Citation Recommendation System",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "rank-bm25>=0.2.2",
        "scikit-learn>=1.3.0",
        "faiss-cpu>=1.7.4",
        "lightgbm>=3.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)

