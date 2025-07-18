## Attributes  

- fixed acidity - Most acids involved with wine that remain stable during fermentation (do not evaporate readily). Acidity is a characteristic determined by the total sum of acids that a sample contains. Fixed acidity corresponds to the set of low volatility organic acids such as malic, lactic, tartaric, or citric acids.  
- volatile acidity - The amount of acetic acids in wine. High levels can lead to an unpleasant, vinegar taste, or even indicate spoilage or bacteria activity. Volatile acidity corresponds to the set of short chain organic acids such as formic, acetic, propionic, and butyric acids.  
- citric acid - A minor acid that adds freshness and flavour to wines. Citric acid is a weak organic acid that occurs naturally in citrus fruits.  
- residual sugar - The amount of sugar remaining after fermentation. Affects a wine’s sweetness.  
- chlorides -  The amount of salt in wine. They are derived from local soil, water, or equipment.  
- free sulfur dioxide - The free form of SO2 that is available to react and exhibits antimicrobial and antioxidant properties. It protects wine from oxidation.  
- total sulfur dioxide - The amount of free and bound forms of SO2. Bound SO2 are those that have reacted with other molecules in the wine.  
- density - Density measures the percent sugar and alcohol content in the wine in relation to pure water. It is used to monitor the fermentation process of the wine. This ratio is called specific gravity.  
- pH - The wine’s acidity level. High pH values in wine have an increased chance of microbial spoilage.  
- sulphates - A wine additive, primarily potassium sulphate, which acts as an antimicrobial to preserve wine from bacteria and yeast-laden invasions.  
- alcohol - Percent alcohol content of the wine.  
- quality - Quality score assigned by wine experts, based on sensory analysis (taste, aroma, mouthfeel, etc). Score between 0-10.  

## Original Dataset Info  

```
RangeIndex: 4898 entries, 0 to 4897
Data columns (total 12 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   fixed_acidity         4898 non-null   float64
 1   volatile_acidity      4898 non-null   float64
 2   citric_acid           4898 non-null   float64
 3   residual_sugar        4898 non-null   float64
 4   chlorides             4898 non-null   float64
 5   free_sulfur_dioxide   4898 non-null   float64
 6   total_sulfur_dioxide  4898 non-null   float64
 7   density               4898 non-null   float64
 8   pH                    4898 non-null   float64
 9   sulphates             4898 non-null   float64
 10  alcohol               4898 non-null   float64
 11  quality               4898 non-null   int64  
dtypes: float64(11), int64(1)
```

The dataset have no null values. It consists of 4898 rows/records and 12 columns/features.

```
       fixed_acidity  volatile_acidity  citric_acid  residual_sugar  \
count    4898.000000       4898.000000  4898.000000     4898.000000   
mean        6.854788          0.278241     0.334192        6.391415   
std         0.843868          0.100795     0.121020        5.072058   
min         3.800000          0.080000     0.000000        0.600000   
25%         6.300000          0.210000     0.270000        1.700000   
50%         6.800000          0.260000     0.320000        5.200000   
75%         7.300000          0.320000     0.390000        9.900000   
max        14.200000          1.100000     1.660000       65.800000   

         chlorides  free_sulfur_dioxide  total_sulfur_dioxide      density  \
count  4898.000000          4898.000000           4898.000000  4898.000000   
mean      0.045772            35.308085            138.360657     0.994027   
std       0.021848            17.007137             42.498065     0.002991   
min       0.009000             2.000000              9.000000     0.987110   
25%       0.036000            23.000000            108.000000     0.991723   
50%       0.043000            34.000000            134.000000     0.993740   
75%       0.050000            46.000000            167.000000     0.996100   
max       0.346000           289.000000            440.000000     1.038980   

                pH    sulphates      alcohol      quality  
count  4898.000000  4898.000000  4898.000000  4898.000000  
mean      3.188267     0.489847    10.514267     5.877909  
std       0.151001     0.114126     1.230621     0.885639  
min       2.720000     0.220000     8.000000     3.000000  
25%       3.090000     0.410000     9.500000     5.000000  
50%       3.180000     0.470000    10.400000     6.000000  
75%       3.280000     0.550000    11.400000     6.000000  
max       3.820000     1.080000    14.200000     9.000000  
```

```
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
Name: quality, dtype: int64  
```

There the quality values range from 3 to 9. There are significantly more normal wines than bad / good wines.

### Data Cleaning
