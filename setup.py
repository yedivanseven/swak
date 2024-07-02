from setuptools import setup, find_packages

PACKAGES = find_packages(exclude=['test*', 'Notebook*'])
NAME = PACKAGES[0]

setup(
    name=NAME,
    description='Toolbox for functional data science projects.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT License',
    license_files=['LICEN[CS]E*'],
    keywords=['functional', 'tools', 'utilities', 'helpers'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.12',
        'Topic :: Utilities',
        'License :: OSI Approved :: MIT License'
    ],
    author='yedivanseven',
    author_email='yedivanseven@icloud.com',
    packages=PACKAGES,
    python_requires='>=3.12',
    install_requires=['pandas>=2.2'],
    setup_requires=['setuptools_scm>=8.1'],
    use_scm_version={
        'version_file': 'version.env',
        'version_file_template': 'SETUPTOOLS_SCM_PRETEND_VERSION={version}'
    }
)
