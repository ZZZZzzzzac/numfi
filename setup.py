import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="numfi", # Replace with your own username
    version="0.2.1",
    author="ZinGer_KyoN",
    author_email="zinger.kyon@gmail.com",
    license='MIT',

    description="a numpy.ndarray subclass that does fixed-point arithmetic",    
    long_description=long_description,
    long_description_content_type="text/markdown",    
    url="https://github.com/ZZZZzzzzac/numfi",
    project_urls={
        "Bug Tracker": "https://github.com/ZZZZzzzzac/numfi/issues",
    },
    
    packages=['numfi'],
    python_requires=">=3.6",
    install_requires=['numpy'],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)