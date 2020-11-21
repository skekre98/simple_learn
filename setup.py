import setuptools
 
with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "numpy"
]
 
setuptools.setup(
    name='simple_learn',  
    version='0.0.1',
    author="Sharvil Kekre",
    python_requires=">=3.6",
    author_email="sharvildev@gmail.com",
    description="A python package to simplify data modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skekre98/simple_learn",
    packages=["simple_learn"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)