# biblio_scikit-learn_poc
## Usage
Usage example:  
```
python simple_classifier.py -d data -f bibliographic_data_small_example.csv -l bib_test.log -e bib_test -v 1 -s Y -m LinearSVC -n model/bib_test
```
Using the created model in python:  
```
python  
>>> from joblib import load  
>>> clf = load('model/bib_test.joblib')  
>>> tst = ["python - how to program", "snow crash", "a guide to chemistry and physics"]  
>>> print(str(clf.predict(tst)))  
>>> exit()  
```
