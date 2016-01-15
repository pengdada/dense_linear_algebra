=======
README
=======

# 1. Code sample name
kmeans

# 2. Description of the code sample package
Note: This application was ported from the Rodinia Suite
      (https://www.cs.virginia.edu/~skadron/wiki/rodinia/).
K-means is a clustering algorithm used extensively in data-mining and elsewhere, important primarily for its simplicity. Many data-mining algorithms show a high degree of data parallelism.
In k-means, a data object is comprised of several values, called features. By dividing a cluster of data objects into K sub-clusters, k-means represents all the data objects by the mean values or centroids of their respective sub-clusters. The initial cluster center for each sub-cluster is randomly chosen or derived from some heuristic. In each iteration, the algorithm associates each data object with its nearest center, based on some chosen distance metric. The new centroids are calculated by taking the mean of all the data objects within each sub-cluster respectively. The algorithm iterates until no data objects move from one sub-cluster to another.

This set of examples demonstrates the use of:
* OpenCL
* NVIDIA CUDA


Additional pre-requisites:
* CUDA
* OpenCL

# 3. Release date
30 July 2015

# 4. Version history 
1.0

# 5. Contributor (s) / Maintainer(s) 
Valeriu Codreanu <valeriu.codreanu@surfsara.nl>

# 6. Copyright / License of the code sample
Apache 2.0

# 7. Language(s) 
C++
CUDA
OpenCL

# 8. Parallelisation Implementation(s)
GPU
CPU

# 9. Level of the code sample complexity 
Source-code example demonstrating the use of CUDA and OpenCL

# 10. Instructions on how to compile the code
Uses the CodeVault CMake infrastructure, see main README.md

# 11. Instructions on how to run the code
Check the kmeans_cuda and kmeans_rodinia_opencl and kmeans_openmp folders for instructions on how to run the application

# 12. Sample input(s)
Input-data is included in the kmeans_data folder

# 13. Sample output(s)
Output cluster center coordinated