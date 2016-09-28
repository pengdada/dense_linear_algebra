=======
README
=======

# 1. Code sample name
pddp2means

# 2. Description of the code sample package
Clustering is the task of grouping a set of objects in such way
that objects assigned in the same group exhibit greater similarity
than those located, according to some computable criterion of
similarity. Our approach on designing a parallel clustering algorithm optimized for the Intel Xeon-Phi
coprocessor.

Our implementation is based on the Principal Direction Divisive Partitioning (PDDP) 2-means algorithm. 

The steps of the implementation include:

1. Create a binary tree.
2. Find a splittable leaf with the greatest scatter value to use pddp 2 means algorithm on.
3. Find an approximation of the dominant eigenvector v of the matrix leaf(data)–leaf(centroid) using the power iteration algorithm.
4. Use the values of vector v to initialize the 2 means algorithm by creating a first set of clusters.
5. Use the result of 2-means to split the cluster of the leaf parent into two new clusters one for each leaf child.
6. Repeat steps 2. to 5. until you have the amount of clusters wanted.

This algorithm provided very stable and accurate solutions for the clustering problem (in terms of the Dunn Index metric).

Additional pre-requisites:
* Intel Compiles Suite
* Intel Xeon Phi

# 3. Release date
25 January 2015

# 4. Version history 
1.0

# 5. Contributor (s) / Maintainer(s) 
Nikos Nikoloutsakos <nikoloutsa@admin.grnet.gr>

# 6. Copyright / License of the code sample
Apache 2.0

    University of Patras, Greece
Copyright (c) 2015 University of Patras
    All rights reserved

   Developed by: HPClab 
Computer Engineering and Informatics Department
    University of Patras

# 7. Language(s) 
C

# 8. Parallelisation Implementation(s)
OpenMP
Intel Xeon Phi - Offload Mode

# 9. Level of the code sample complexity 
Advanced.

# 10. Instructions on how to compile the code
Please use intel compilers to enable Xeon Phi Offload mode
CC=icc cmake ..

# 11. Instructions on how to run the code
Usage:
./<exe> <input_file> <output_file> <clusters>

Sample input example:
./pddp_2means_omp ../data/40k.csv pddp_2means.out 19

# 12. Sample input(s)
Input-data is included in the data/ folder

# 13. Sample output(s)
Output the membership of the data point to the cluster.
