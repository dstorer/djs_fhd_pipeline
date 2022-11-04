import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='djs_fhd_pipeline',
    version='0.0.0',
    author='Dara Storer',
    author_email='darajstorer@gmail.com',
    description='Tools for processing HERA data with FHD',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/dstorer/djs_fhd_pipeline',
    license='MIT',
    packages=['djs_fhd_pipeline'],
    install_requires=['numpy>=1.18',
                      'matplotlib',
                      'pyuvdata'],
)