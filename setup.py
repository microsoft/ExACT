from setuptools import setup, find_packages


setup(
    name="exact",
    version="0.0.1",
    author="Xiao Yu",
    author_email="xy2437@columbia.edu",
    description="",
    long_description=open('README.md', encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(include=["exact"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        # in addition to installing OSWorld
        "jsonlines",
        "text_generation",
        "graphviz",
        "langchain",
        "langchain_community",
        "langchain_openai",
        "faiss-gpu",
        "fastapi",
        "uvicorn",
    ],
    cmdclass={}
)