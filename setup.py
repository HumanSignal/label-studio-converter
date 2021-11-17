import setuptools

# Readme
with open('README.md', 'r') as f:
    long_description = f.read()

# Module dependencies
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
	
with open('label_studio_converter/__init__.py') as f:
    version = re.search("__version__ ?= ?'(.*?)'", f.read()).group(1)

setuptools.setup(
    name='label-studio-converter',
    version=version,
    author='Heartex',
    author_email="hello@heartex.ai",
    description='Format converter add-on for Label Studio',
    long_description='',
    long_description_content_type='text/markdown',
    url='https://github.com/heartexlabs/label-studio-converter',
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=requirements,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'label-studio-converter=label_studio_converter.main:main',
        ],
    }
)
