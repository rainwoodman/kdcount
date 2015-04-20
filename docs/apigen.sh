# bash

if ! python -c 'import numpydoc'; then easy_install --user numpydoc; fi
if ! python -c 'import sphinx'; then easy_install --user sphinx; fi
pushd ..
python setup.py build_ext --inplace
popd
sphinx-apidoc -H "API Reference" -e -f -o . ../kdcount
