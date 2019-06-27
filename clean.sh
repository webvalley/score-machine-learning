jupyter nbconvert --ClearOutputPreprocessor.enabled=True --clear-output $(find . | grep -e ".ipynb$")

