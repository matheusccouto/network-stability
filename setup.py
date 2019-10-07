import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="network_stability",
    version="1.0.1",
    author="Matheus Couto",
    author_email="matheusccouto@gmail.com",
    description="Network connectivity and speed test.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matheusccouto/network-stability",
    packages=setuptools.find_packages(),
    classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                ],
    python_requires='>=3.6',
    install_requires=['numpy', 'pandas', 'matplotlib', 'speedtest-cli', 'scipy'],
    keywords = ['NETWORK', 'STABILITY', 'TEST', 'SPEED', 'CONNECTION'],
)