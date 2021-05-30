# PolyGraphs

## Summary

## Quick start

```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ python setup.py install
$ python -m dgl.backend.set_default_backend pytorch
```

## Google Colaboratory:

```
!git clone https://[token]@github.com/alexandroskoliousis/polygraphs.git
%cd polygraphs
!python setup.py install
!nvidia-smi
!pip install dgl-cu110
```