### Table 1: Comprehensive Dataset Statistics & Length Distributions

| Dataset                 | Domain / Type           |   Total Samples |   Fake Count | Fake %   |   Real Count | Real %   |   Train Set (80%) |   Test Set (20%) | Avg Words (Raw)   | Avg Words (Clean)   | Min/Max Words   |
|:------------------------|:------------------------|----------------:|-------------:|:---------|-------------:|:---------|------------------:|-----------------:|:------------------|:--------------------|:----------------|
| ISOT (Our Cleaned)      | Long-form News Articles |           39100 |        17903 | 45.79%   |        21197 | 54.21%   |             31280 |             7820 | 410.5             | 239.9               | 3/5008          |
| ISOT (Paper Raw)        | Long-form News Articles |           44898 |        23481 | 52.30%   |        21417 | 47.70%   |             35918 |             8980 | ~410              | N/A                 | N/A             |
| LIAR (Our Binary Split) | Short Claim Statements  |           12791 |         5657 | 44.23%   |         7134 | 55.77%   |             10232 |             2559 | 18.0              | 17.2                | 2/437           |
| LIAR (Wang 2017 TSV)    | Short Claim Statements  |           12791 |         5657 | 44.23%   |         7134 | 55.77%   |             10240 |             2551 | 18.0              | N/A                 | N/A             |
