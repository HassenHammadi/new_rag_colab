from setuptools import setup, find_packages

setup(
    name="new_rag_colab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.5.2",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "ipywidgets>=7.7.1",
        "pypdf>=4.0.0",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "numpy>=1.24.3",
        "tqdm>=4.66.1",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="RAG System for Google Colab with Drive Integration",
    keywords="rag, colab, google drive, nlp",
    url="https://github.com/yourusername/new_rag_colab",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)
