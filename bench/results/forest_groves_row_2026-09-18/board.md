| model | rows | columns | outputs | 32-thread kernels (default) ms | row schedule ms | ratio | hashes | sabotage hash |
|---|---|---|---|---|---|---|---|---|
| rf-taxi-100x16 | 4,610,786 | 16 | 2 | 189.4 / 190.7 | 125.7 / 127.5 | 1.49 | equal | differs |
| rf-taxireg-100x16 | 5,750,086 | 16 | 1 | 237.9 / 239.0 | 157.4 / 158.1 | 1.51 | equal | differs |
| rf-higgs-100x16 | 500,000 | 28 | 2 | 24.0 / 24.3 | 20.4 / 19.9 | 1.19 | equal | differs |
| et-higgs-100x16 | 500,000 | 28 | 2 | 66.5 / 67.4 | 68.0u / 63.5 | 0.99 | equal | differs |
| rf-higgs-500x16 | 500,000 | 28 | 2 | 137.6 / 138.6 | 104.0 / 105.1 | 1.32 | equal | differs |
| rf-covtype-100x16 | 581,012 | 54 | 7 | 27.0u / 27.4 | 32.0 / 32.5u | 0.84 | equal | differs |
| et-year-100x16 | 515,345 | 90 | 1 | 56.5 / 54.1 | 63.8 / 64.9 | 0.87 | equal | differs |
| rf-istella-100x16 | 2,543,304 | 220 | 2 | 275.2 / 273.8 | 619.4 / 616.4 | 0.44 | equal | differs |
| rf-istellareg-100x16 | 2,543,304 | 220 | 1 | 263.1 / 257.5 | 545.0 / 546.3 | 0.48 | equal | differs |

geomean all nine 0.94; the dispatch takes the row schedule only for columns <= 32
