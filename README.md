# Multi-Parallel Feature Selection Genetic Algorithm

This is a zero external-dependencie genetic algorithm for `.arff` hierarchy optimization. You can use open-source datasets from [my blog](https://oliveira-sh.github.io/datasets/hierarchical/).

100% compatibility with pypy for faster code generation.

### File

```
@relation 'your_datasets'

@attribute A                    {0,1,2,3,4}   # Should be in a discret interval
@attribute B                    {0,1,2,3,4}
@attribute C                    {0,1,2,3,4}
@attribute class {01,01.1,01.2}               # Your classes
@data
0,3,4,01                                      # Your @data, class.
2,2,3,01.1
4,4,01.2
```

### Usage

```
usage: main.py [-h] --train TRAIN [--pop POP] [--gen GEN] [--cxpb CXPB] [--mutpb MUTPB] [--mlnp] [--usf]
               [--out OUT]

options:
  -h, --help     show this help message and exit
  --train TRAIN  ARFF used for 5-fold CV
  --pop POP
  --gen GEN
  --cxpb CXPB
  --mutpb MUTPB
  --mlnp         flag mandatory leaf nodes
  --usf          use log-usefulness
  --out OUT      output folder for final ARFF
```
