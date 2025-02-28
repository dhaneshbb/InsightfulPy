from setuptools import setup, find_packages

setup(
    name="insightfulpy",  
    version="0.1.7",
    author="dhaneshbb",
    author_email='dhaneshbb5@gmail.com',
    description="A toolkit for insightful exploratory data analysis (EDA) with advanced visualization and statistical tools.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dhaneshbb/insightfulpy", 
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "pandas",
        "researchpy",
        "tabulate",
        "tableone",
        "matplotlib",
        "missingno",
        "seaborn",
        "scipy",
        "numpy"
    ],
    license="MIT",
    include_package_data=True,
)
