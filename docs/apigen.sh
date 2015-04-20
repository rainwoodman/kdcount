# bash

if ! python -c 'import numpydoc'; then easy_install --user numpydoc; fi
if ! python -c 'import sphinx'; then easy_install --user sphinx; fi
pushd ..
python setup.py build_ext --inplace
sphinx-apidoc -e -f -o . ../kdcount
popd
