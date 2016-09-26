#include <omp.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <alloca.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>

/*
 * Each line of the input data file must be less than this value.
 */
#define BUFFER_SIZE			(1024*1024)

/*
 * Define the alignment in bytes that is required for optimal vectorization
 */
#define ALIGNMENT			64

#define TWO_MEANS_TOLERANCE		1.0e-4
#define TWO_MEANS_MAX_ITERATIONS	50
#define POWER_ITERATION_TOLERANCE	1.0e-8
#define POWER_ITERATION_MAX_ITERATIONS	5000

//#define DEBUG	1

#if defined(DEBUG)
#define DBG(format, ...)	printf(format, __VA_ARGS__)
#else
#define DBG(format, ...)
#endif

struct node {
	unsigned long	*indices;	/* Number of data_points in the cluster			(data_points) */
	double		*centroid;	/* The mean vector for the cluster			(attributes)  */
	struct node	*leftchild;
	struct node	*rightchild;
	struct node	*sibling;
	double		scat;		/* Total scatter value for the cluster		*/
	unsigned long	num_of_indices;	/* Number of elements in vector 'indices'	*/
	unsigned long	keep_in_tree;
	unsigned long	is_splittable;
};

typedef struct node	Node;

/*******************************************************************************/

#pragma offload_attribute (push,target(mic))

void print_elapsed_time(struct timeval start, struct timeval end, char *msg);
Node *allocate_node(unsigned long data_points);
void init_node(Node *node, Node *sibling, double *M, unsigned long data_points, unsigned long *indices, double *centroid, unsigned long attributes, unsigned long padded_attributes, unsigned long threads);
void process_node(Node *node, double *M, unsigned long attributes, unsigned long padded_attributes, unsigned long current_level, unsigned long max_level);
double calc_scatter(Node *node, double *M, unsigned long attributes, unsigned long padded_attributes, unsigned long threads);
void vector_2_means(Node *node, double *M, double *v, unsigned long attributes, unsigned long padded_attributes, unsigned long **l_indices, unsigned long **r_indices, double **l_centroid, double **r_centroid, unsigned long *l_data_points, unsigned long *r_data_points, unsigned long threads);
void power_iteration(Node *node, double *M, unsigned long attributes, unsigned long padded_attributes, double **v, unsigned long threads);
void reset_nodes_to_keep(Node *node);
void keep_all_nodes(Node *node);
void find_max_scatter_value(Node *node, Node **K);
unsigned long find_nodes_to_keep(Node *node, unsigned long clusters);
double max_cluster_diameter(Node *node, double *M, unsigned long attributes, unsigned long padded_attributes);
void find_all_leaves(Node *node, Node **cluster_nodes, unsigned long *cluster_num);
void find_all_splittable_leaves(Node *node, Node **cluster_nodes, unsigned long *cluster_num);
double min_cluster_distance(Node *node, unsigned long clusters, unsigned long attributes);
void post_order_traversal(Node *node, unsigned long *indices, unsigned long *cluster);
void assign_cluster_numbers(Node *node, unsigned long *result, unsigned long data_points);

#pragma offload_attribute (pop)

void write_results(unsigned long *result, unsigned long data_points, char *filename);

/*******************************************************************************/

