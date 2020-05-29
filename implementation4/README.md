This project explores k-means clustering and dimension reduciton (PCA) algorithms for identifying what activity (standing, running, sitting, etc...) a Sumsung Galaxy S3 phone detects its user is participating in based off of its accelerometer and gyros data. Starter code was provided by Hamed Shahbazi.
To run:

```bash
cd src
python3 main.py
```

Code for k-means clustering can be found in src/clustering.py. Code for dimension reduction can be found in src/decompose.py. src/main.py includes functions for generating plots to visualize the algorithms. 
Code can also be run for specific classifier generation and testing ->

K-Means Clustering:
```bash
python3 main.py --kmeans=1 --pca=0
```

Dimension Reduction (PCA):
```bash
python3 main.py --pca=1
```

