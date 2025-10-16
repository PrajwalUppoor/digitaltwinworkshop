"""Try loading smart_home_model.pkl with different loaders and report results.

Run from the repo root: python tools\try_load_model.py
"""
import pickle
import traceback
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "smart_home_model.pkl"
print("Trying to load:", p)
loaders = []
try:
    import joblib
    loaders.append(('joblib', lambda path: joblib.load(path)))
except Exception:
    print('joblib not available')
try:
    import cloudpickle
    loaders.append(('cloudpickle', lambda path: cloudpickle.load(open(path, 'rb'))))
except Exception:
    print('cloudpickle not available')
loaders.insert(0, ('pickle', lambda path: pickle.load(open(path, 'rb'))))
loaders.append(('pickle_latin1', lambda path: pickle.load(open(path, 'rb'), encoding='latin1')))

for name, fn in loaders:
    try:
        print(f'--- Trying {name} ---')
        obj = fn(p)
        print(f'{name} succeeded: type={type(obj)}')
        break
    except Exception as e:
        print(f'{name} failed: {e}')
        traceback.print_exc()

print('done')
