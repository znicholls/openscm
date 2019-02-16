"""OpenSCM
"""

import versioneer

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

PACKAGE_NAME = "openscm"
AUTHOR = "Robert Gieseke"
EMAIL = "robert.gieseke@pik-potsdam.de"
URL = "https://github.com/openclimatedata/openscm"

DESCRIPTION = "A unifying interface for Simple Climate Models"
README = "README.rst"

REQUIREMENTS = [
    "numpy",
    "pint",
    "pandas",
    # can be moved into notebooks dependencies once Jared's new backend is in place
    "pyam-iamc @ git+https://github.com/IAMconsortium/pyam.git@master",
]
REQUIREMENTS_NOTEBOOKS = ["notebook", "seaborn"]
REQUIREMENTS_TESTS = ["codecov", "nbval", "pytest", "pytest-cov"]
REQUIREMENTS_DOCS = ["sphinx>=1.4", "sphinx_rtd_theme", "sphinx-autodoc-typehints"]
REQUIREMENTS_DEPLOY = ["setuptools>=38.6.0", "twine>=1.11.0", "wheel>=0.31.0"]

requirements_dev = [
    *["flake8", "black"],
    *REQUIREMENTS_NOTEBOOKS,
    *REQUIREMENTS_TESTS,
    *REQUIREMENTS_DOCS,
    *REQUIREMENTS_DEPLOY,
]

requirements_extras = {
    "notebooks": REQUIREMENTS_NOTEBOOKS,
    "docs": REQUIREMENTS_DOCS,
    "tests": REQUIREMENTS_TESTS,
    "deploy": REQUIREMENTS_DEPLOY,
    "dev": requirements_dev,
}

REQUIREMENTS_MODELS = {}
"""
When implementing an additional adapter, include your adapter NAME here as:
```
"NAME": [ ... additional pip modules you need ... ],
```
"""

for k, v in REQUIREMENTS_MODELS.items():
    requirements_extras["model-{}".format(k)] = v
    REQUIREMENTS_TESTS += v

# Get the long description from the README file
with open(README, "r", encoding="utf-8") as f:
    README_TEXT = f.read()


class OpenSCMTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        pytest.main(self.test_args)


cmdclass = versioneer.get_cmdclass()
cmdclass.update({"test": OpenSCMTest})

setup(
    name=PACKAGE_NAME,
    version=versioneer.get_version(),
    description=DESCRIPTION,
    long_description=README_TEXT,
    long_description_content_type="text/x-rst",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license="GNU Affero General Public License v3.0 or later",
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    ],
    keywords=["simple climate model"],
    packages=find_packages(exclude=["tests"]),
    install_requires=REQUIREMENTS,
    extras_require=requirements_extras,
    cmdclass=cmdclass,
    project_urls={
        "Bug Reports": "https://github.com/openclimatedata/openscm/issues",
        "Source": "https://github.com/openclimatedata/openscm/",
    },
)
