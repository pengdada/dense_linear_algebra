#include "pddp_2means.h"

/******************************************************************************/

#pragma offload_attribute (push, target(mic))

/******************************************************************************/

volatile unsigned long	leaves = 1, active_data_points;
unsigned long		num_of_cores;

/******************************************************************************/

void print_elapsed_time(struct timeval start, struct timeval end, char *msg)
{
	printf("%s: %f sec\n", msg, ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / 1000000.0 );
	fflush(NULL);
}

/******************************************************************************/

Node *allocate_node(unsigned long data_points)
{
	Node	*temp;
	int	error;

	error = posix_memalign((void **)(&temp), 64, sizeof(Node));

	if (error != 0) {
		printf("ERROR: Could not allocate memory for tree node.\n");
		exit(0);
	}

	return(temp);
}

/******************************************************************************/

void init_node(Node *node, Node *sibling, double *M, unsigned long data_points, unsigned long *indices, double *centroid, unsigned long attributes, unsigned long padded_attributes, unsigned long threads)
{
	node->num_of_indices 	= data_points;
	node->indices		= indices;
	node->centroid		= centroid;
	node->leftchild		= NULL;
	node->rightchild	= NULL;
	node->sibling		= sibling;
	node->keep_in_tree	= 0;
	node->is_splittable  	= 1;
	node->scat		= calc_scatter(node, M, attributes, padded_attributes, threads);
}

/******************************************************************************/

void process_node(Node *node, double *M, unsigned long attributes, unsigned long padded_attributes, unsigned long current_level, unsigned long max_level)
{
	unsigned long	l_data_points, r_data_points, *l_indices, *r_indices, threads;
	double		*l_centroid, *r_centroid, *v;

	#if defined(_OPENMP)
	omp_set_nested(1);
	#endif

	threads = lround(((double)(node->num_of_indices * num_of_cores)) / ((double)active_data_points));
	if (threads < 1) {
		threads = 1;
	}

	/*
	 * Calculate leading principal component using the power iteration method.
	 */
	power_iteration(node, M, attributes, padded_attributes, &v, threads);

	/*
	 * Split cluster into 2 new clusters using the 2-means algorithm.
	 */
	vector_2_means(node, M, v, attributes, padded_attributes, &l_indices, &r_indices, &l_centroid, &r_centroid, &l_data_points, &r_data_points, threads);

	munmap(v, node->num_of_indices * sizeof(double));

	/*
	 * Check that both new clusters will contain enough data points.
	 */
	if ((l_data_points > 1) && (r_data_points > 1)) {

		node->leftchild  = allocate_node(l_data_points);
		node->rightchild = allocate_node(r_data_points);
		DBG("Left  node allocated is %p\n", node->leftchild);
		DBG("Right node allocated is %p\n", node->rightchild);

		init_node(node->leftchild, node->rightchild, M, l_data_points, l_indices, l_centroid, attributes, padded_attributes, threads);
		init_node(node->rightchild, node->leftchild, M, r_data_points, r_indices, r_centroid, attributes, padded_attributes, threads);
		DBG("Left  child has %ld indices\n", node->leftchild->num_of_indices);
		DBG("Right child has %ld indices\n", node->rightchild->num_of_indices);

		#pragma omp atomic
		leaves++;

		if (current_level < max_level - 1) {
			#pragma omp task
			process_node(node->leftchild,  M, attributes, padded_attributes, current_level + 1, max_level);

			#pragma omp task
			process_node(node->rightchild, M, attributes, padded_attributes, current_level + 1, max_level);
		} else {
			#pragma omp atomic
			active_data_points -= node->num_of_indices;
		}
	} else {
		#pragma omp atomic
		active_data_points -= node->num_of_indices;

		node->is_splittable = 0;

		if (l_indices != NULL) {
			munmap(l_indices, l_data_points * sizeof(unsigned long));
		}

		if (r_indices != NULL) {
			munmap(r_indices, r_data_points * sizeof(unsigned long));
		}
		free(l_centroid);
		free(r_centroid);
	}
}

/******************************************************************************/

double calc_scatter(Node *node, double *M, unsigned long attributes, unsigned long padded_attributes, unsigned long threads)
{
	unsigned long	i, j;
	double		scat = 0.0, *M_line;

	#pragma omp parallel for default(none) private(i, j, M_line) shared(node, M, attributes, padded_attributes) reduction(+:scat) num_threads(threads)
	for (i = 0; i < node->num_of_indices; i++) {
		M_line = &M[node->indices[i] * padded_attributes];
		#pragma vector aligned
		#pragma ivdep
		for (j = 0; j < attributes; j++) {
			scat += M_line[j] * M_line[j];
		}
	}

	DBG("Scatter value for node %p is %f\n", node, scat);

	return(scat);
}

/******************************************************************************/

void vector_2_means(Node *node, double *M, double *v, unsigned long attributes, unsigned long padded_attributes, unsigned long **l_indices, unsigned long **r_indices,
		    double **l_centroid, double **r_centroid, unsigned long *l_data_points, unsigned long *r_data_points, unsigned long threads)
{
	double		*c_l, *c_r, *M_line;
	double		prev_diff, diff = INFINITY;
	unsigned long	i, j, iter, l = 0, r = 0, error, *i_l, *i_r;
	char		*clusters;

	/*
	 * Allocate memory for centroids of the two clusters to be created.
	 */
	error = posix_memalign((void **)(&c_l), 64, attributes * sizeof(double));
	if (error != 0) {
		printf("ERROR: Could not allocate memory for vector c_l.\n");
		exit(0);
	}

	error = posix_memalign((void **)(&c_r), 64, attributes * sizeof(double));
	if (error != 0) {
		printf("ERROR: Could not allocate memory for vector c_r.\n");
		exit(0);
	}

	clusters = (char *)alloca(node->num_of_indices * sizeof(char));

	__assume_aligned(node, 64);
	__assume_aligned(v, 64);
	__assume_aligned(*l_indices, 64);
	__assume_aligned(*r_indices, 64);
	__assume_aligned(*l_centroid, 64);
	__assume_aligned(*r_centroid, 64);
	__assume_aligned(l_data_points, 64);
	__assume_aligned(r_data_points, 64);
//	__assume_aligned(clusters, 64);
	__assume_aligned(M_line, 64);
	__assume_aligned(c_l, 64);
	__assume_aligned(c_r, 64);
	__assume_aligned(i_l, 64);
	__assume_aligned(i_r, 64);

	/*
	 * Initialize centroids.
	 */
	memset(c_l, 0, attributes * sizeof(double));
	memset(c_r, 0, attributes * sizeof(double));

	/*
	 * Make an initial assignment of each data point to a cluster and
	 * calculate the initial centroids of the two clusters.
	 */
	#pragma omp parallel default(none) private(i, j, M_line) shared(l, r, node, clusters, c_l, c_r, v, M, attributes, padded_attributes) num_threads(threads)
	{
	__attribute__((aligned(64))) double	c_l_local[attributes], c_r_local[attributes];

	memset(c_l_local, 0, attributes * sizeof(double));
	memset(c_r_local, 0, attributes * sizeof(double));

	#pragma omp for reduction(+:l, r)
	for (i = 0; i < node->num_of_indices; i++) {
		M_line = &M[node->indices[i] * padded_attributes];
		if (v[i] <= 0.0) {
			#pragma vector aligned
			#pragma ivdep
			for (j = 0; j < attributes; j++) {
				c_l_local[j] += M_line[j];
			}
			clusters[i] = 0;
			l++;
		} else {
			#pragma vector aligned
			#pragma ivdep
			for (j = 0; j < attributes; j++) {
				c_r_local[j] += M_line[j];
			}
			clusters[i] = 1;
			r++;
		}
	}

	#pragma vector aligned
	#pragma ivdep
	for (j = 0; j < attributes; j++) {
		#pragma omp atomic
		c_l[j] += c_l_local[j];
		#pragma omp atomic
		c_r[j] += c_r_local[j];
	}
	}

	#pragma vector aligned
	#pragma ivdep
	for (i = 0; i < attributes; i++) {
		c_l[i] /= l;
		c_r[i] /= r;
	}

	/*
	 * Apply the 2-means algorithm to refine the assignment of data points to the two clusters.
	 */
	iter = 0;
	do {
		prev_diff = diff;
		diff = 0.0;

		#pragma omp parallel default(none) private(i, j, M_line) shared(l, r, node, clusters, c_l, c_r, M, attributes, padded_attributes) reduction(+:diff) num_threads(threads)
		{
		__attribute__((aligned(64))) double	dist1, dist2;
		__attribute__((aligned(64))) double	c_l_local[attributes], c_r_local[attributes];

		memset(c_l_local, 0, attributes * sizeof(double));
		memset(c_r_local, 0, attributes * sizeof(double));

		/*
		 * For every data point of the node we are processing.
		 */
		#pragma omp for schedule(static) reduction(+:l, r) nowait
		for (i = 0; i < node->num_of_indices; i++) {

			dist1 = 0.0;
			dist2 = 0.0;

			/*
			 * Calculate the Euclidean distance from the data point to each of the two cluster centers.
			 */
			M_line = &M[node->indices[i] * padded_attributes];
			#pragma vector aligned
			#pragma ivdep
			for (j = 0; j < attributes; j++) {
				dist1 += (c_l[j] - M_line[j]) * (c_l[j] - M_line[j]);
				dist2 += (c_r[j] - M_line[j]) * (c_r[j] - M_line[j]);
			}

			/*
			 * If the data point is closer to the left-side cluster center, then assign it to that cluster.
			 * Furthermore, check whether the data point is changing cluster (it previously belonged to the right-side cluster)
			 * and update accordingly the number of data points that belong to each cluster.
			 */
			if (dist1 < dist2) {
				if (clusters[i] == 1) {
					l++;
					r--;
				}
				clusters[i] = 0;
				diff += dist1;
			} else {
				if (clusters[i] == 0) {
					l--;
					r++;
				}
				clusters[i] = 1;
				diff += dist2;
			}
		}

		/*
		 * Calculate new centroids.
		 */
		#pragma omp for schedule(static)
		for (i = 0; i < node->num_of_indices; i++) {
			M_line = &M[node->indices[i] * padded_attributes];
			if (clusters[i] == 0) {
				#pragma vector aligned
				#pragma ivdep
				for (j = 0; j < attributes; j++) {
					c_l_local[j] += M_line[j];
				}
			} else {
				#pragma vector aligned
				#pragma ivdep
				for (j = 0; j < attributes; j++) {
					c_r_local[j] += M_line[j];
				}
			}
		}

		#pragma omp single
		{
		memset(c_l, 0, attributes * sizeof(double));
		memset(c_r, 0, attributes * sizeof(double));
		}

		#pragma vector aligned
		#pragma ivdep
		for (j = 0; j < attributes; j++) {
			#pragma omp atomic
			c_l[j] += c_l_local[j];
			#pragma omp atomic
			c_r[j] += c_r_local[j];
		}
		}

		#pragma vector aligned
		#pragma ivdep
		for (i = 0; i < attributes; i++) {
			c_l[i] /= l;
			c_r[i] /= r;
		}

		iter++;
//		printf("%p prev_diff = %.20f, diff = %.20f, fabs(diff - prev_diff) = %.20f\n", node, prev_diff, diff, fabs(diff - prev_diff));
	} while ((fabs(diff - prev_diff) > TWO_MEANS_TOLERANCE) && (iter < TWO_MEANS_MAX_ITERATIONS));

//	printf("2-means ended after %ld iterations.\n", iter);

	/*
	 * Return to the caller the calculated size of each cluster.
	 */
	*l_data_points = l;
	*r_data_points = r;

	/*
	 * Return to the caller the calculated centroid of each cluster.
	 */
	*l_centroid = c_l;
	*r_centroid = c_r;

	if ((l > 0) && (r > 0)) {
		i_l = (unsigned long *)mmap(NULL, l * sizeof(unsigned long), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
		i_r = (unsigned long *)mmap(NULL, r * sizeof(unsigned long), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

		if ((i_l == MAP_FAILED) || (i_r == MAP_FAILED)) {
			printf("ERROR: Could not allocate memory for a vector in 2-means.\n");
			exit(0);
		}

		/*
		 * Find to which cluster each data point belongs.
		 */
		l = 0;
		r = 0;
		#pragma vector aligned
		#pragma ivdep
		for (i = 0; i < node->num_of_indices; i++) {
			if (clusters[i] == 0) {
				i_l[l] = node->indices[i];
				l++;
			} else {
				i_r[r] = node->indices[i];
				r++;
			}
		}

		/*
		 * Return to the caller the indices that belong to each cluster.
		 */
		*l_indices = i_l;
		*r_indices = i_r;
	} else if (l == 0) {
		i_r = (unsigned long *)mmap(NULL, r * sizeof(unsigned long), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

		if (i_r == MAP_FAILED) {
			printf("ERROR: Could not allocate memory for right vector in 2-means.\n");
			exit(0);
		}

		/*
		 * All data points belong to the right cluster.
		 */
		#pragma vector aligned
		#pragma ivdep
		for (i = 0; i < node->num_of_indices; i++) {
			i_r[i] = node->indices[i];
		}

		/*
		 * Return to the caller the indices that belong to each cluster.
		 */
		*l_indices = NULL;
		*r_indices = i_r;
	
	} else {
		i_l = (unsigned long *)mmap(NULL, l * sizeof(unsigned long), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

		if (i_l == MAP_FAILED) {
			printf("ERROR: Could not allocate memory for left vector in 2-means.\n");
			exit(0);
		}

		/*
		 * All data points belong to the left cluster.
		 */
		#pragma vector aligned
		#pragma ivdep
		for (i = 0; i < node->num_of_indices; i++) {
			i_l[i] = node->indices[i];
		}

		/*
		 * Return to the caller the indices that belong to each cluster.
		 */
		*l_indices = i_l;
		*r_indices = NULL;

	}
}

/******************************************************************************/

void power_iteration(Node *node, double *M, unsigned long attributes, unsigned long padded_attributes, double **v, unsigned long threads)
{
	double		*x_prev, *x_curr, *x_temp, max, norm = INFINITY;
	unsigned long	i, j;

	x_prev = (double *)mmap(NULL, node->num_of_indices * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	x_curr = (double *)mmap(NULL, node->num_of_indices * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	x_temp = (double *)mmap(NULL,           attributes * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

	if ((x_prev == MAP_FAILED) || (x_curr == MAP_FAILED) || (x_temp == MAP_FAILED)) {
		printf("ERROR: Could not allocate a vector in power iteration.\n");
		exit(0);
	}

	__assume_aligned(x_prev, 64);
	__assume_aligned(x_curr, 64);
	__assume_aligned(x_temp, 64);
	__assume_aligned(node, 64);
	__assume_aligned(*v, 64);

	#pragma vector aligned
	#pragma ivdep
	for (i = 0; i < node->num_of_indices; i++) {
		x_curr[i] = 1.0;
	}

	#pragma omp parallel default(none) private(i, j) shared(x_temp, x_curr, x_prev, node, max, norm, M, attributes, padded_attributes) num_threads(threads) if(threads > 1)
	{
	__attribute__((aligned(64))) double	x_temp_local[attributes];

	double		*temp, *M_line;
	unsigned long	iter = 0;

	__assume_aligned(M_line, 64);

	do {
		#pragma omp single
		{
		max  = 0.0;

		temp   = x_prev;
		x_prev = x_curr;
		x_curr = temp;

		memset(x_temp, 0, attributes * sizeof(double));
		}

		memset(x_temp_local, 0, attributes * sizeof(double));

		#pragma omp for nowait
		for (i = 0; i < node->num_of_indices; i++) {
			M_line = &M[node->indices[i] * padded_attributes];
			#pragma vector aligned nontemporal
			#pragma ivdep
			for (j = 0; j < attributes; j++) {
				x_temp_local[j] += (M_line[j] - node->centroid[j]) * x_prev[i];
			}
		}

		#pragma vector aligned
		#pragma ivdep
		for (j = 0; j < attributes; j++) {
			#pragma omp atomic
			x_temp[j] += x_temp_local[j];
		}

		norm = 0.0;
		#pragma omp barrier

		#pragma omp for reduction(max:max)
		for (i = 0; i < node->num_of_indices; i++) {
			x_curr[i] = 0.0;
			M_line = &M[node->indices[i] * padded_attributes];
			#pragma vector aligned nontemporal
			#pragma ivdep
			for (j = 0; j < attributes; j++) {
				x_curr[i] += (M_line[j]  - node->centroid[j]) * x_temp[j];
			}

			if (fabs(x_curr[i]) > max) {
				max = fabs(x_curr[i]);
			}
		}

		#pragma omp for reduction(+:norm)
		for (i = 0; i < node->num_of_indices; i++) {
			x_curr[i] /= max;
			norm += (x_curr[i] - x_prev[i]) * (x_curr[i] - x_prev[i]);
		}

		iter++;

	} while (((norm / node->num_of_indices) > POWER_ITERATION_TOLERANCE) && (iter < POWER_ITERATION_MAX_ITERATIONS));
	}

	munmap(x_prev, node->num_of_indices * sizeof(double));
	munmap(x_temp, attributes * sizeof(double));

	*v = x_curr;
}

/******************************************************************************/

void reset_nodes_to_keep(Node *node)
{
	if (node != NULL) {
		reset_nodes_to_keep(node->leftchild);
		reset_nodes_to_keep(node->rightchild);

		node->keep_in_tree = 0;
	}
}

/******************************************************************************/

void keep_all_nodes(Node *node)
{
	if (node != NULL) {
		keep_all_nodes(node->leftchild);
		keep_all_nodes(node->rightchild);

		node->keep_in_tree = 2;
	}
}

/******************************************************************************/

void find_max_scatter_value(Node *node, Node **K)
{
	if (node != NULL) {
		find_max_scatter_value(node->leftchild,  K);
		find_max_scatter_value(node->rightchild, K);

		if ((node->scat > (*K)->scat) && (node->keep_in_tree < 2)) {
			(*K) = node;
		}
	}
}

/******************************************************************************/

unsigned long find_nodes_to_keep(Node *node, unsigned long clusters)
{
	unsigned long	c = 1;
	Node		*K, dummy;

	dummy.scat = -INFINITY;

	/*
	 * The meaning of the values for 'keep_in_tree' is as follows:
	 * - 0 means that the node is not in the tree.
	 * - 1 means that the node is in the tree because its parent had at some point
	 *   the largest scatter value among the nodes that had not been added to the
	 *   tree yet (hence it was split and its children must belong to the tree).
	 *   However, there is a possibility that it's scatter value will be checked later on
	 *   and if it is the largest among the remaining nodes further processing will happen
	 *   for the node itself and its children (if it has children).
	 * - 2 means that the node is in the tree and no further checks will be done.
	 */
	node->keep_in_tree = 1;
	do {
		K = &dummy;
		find_max_scatter_value(node, &K);

		K->keep_in_tree = 2;
		if (K->leftchild != K->rightchild) {
			K->leftchild->keep_in_tree  = 1;
			K->rightchild->keep_in_tree = 1;
			c += 2;
		} else if (K->is_splittable == 1) {
			reset_nodes_to_keep(node);
			return(0L);
		}
	} while (c < (2 * clusters - 1));

	return(1L);
}

/******************************************************************************/

double max_cluster_diameter(Node *node, double *M, unsigned long attributes, unsigned long padded_attributes)
{
	double	l_max_dist = 0.0, r_max_dist = 0.0, max_dist = 0.0, *M_line_i, *M_line_j;

	__assume_aligned(node, 64);
	__assume_aligned(M, 64);
	__assume_aligned(M_line_i, 64);
	__assume_aligned(M_line_j, 64);

	if (node != NULL) {
		l_max_dist = max_cluster_diameter(node->leftchild,  M, attributes, padded_attributes);
		r_max_dist = max_cluster_diameter(node->rightchild, M, attributes, padded_attributes);

		if (l_max_dist > r_max_dist) {
			max_dist = l_max_dist;
		} else {
			max_dist = r_max_dist;
		}

		/*
		 * The PDDP algorithm ensures that every node has either 2 children or is a leaf node.
		 * Hence, a leaf node is found when both pointers to the children are equal (both are NULL).
		 */
		if (((node->keep_in_tree == 1) || (node->keep_in_tree == 2)) && ((node->leftchild == node->rightchild) || (node->leftchild->keep_in_tree == 0) || (node->rightchild->keep_in_tree == 0))) {
			unsigned long	i, j, k;
			double		dist;

			#pragma omp parallel for default(none) private(i, j, k, dist, M_line_i, M_line_j) shared(node, M, attributes, padded_attributes) reduction(max:max_dist) num_threads(omp_get_num_procs())
			for (i = 0; i < node->num_of_indices; i++) {
				M_line_i = &M[node->indices[i] * padded_attributes];
				for (j = i + 1; j < node->num_of_indices; j++) {
					M_line_j = &M[node->indices[j] * padded_attributes];
					dist = 0.0;
					#pragma vector aligned
					#pragma ivdep
					for (k = 0; k < attributes; k++) {
						dist += (M_line_i[k] - M_line_j[k]) * (M_line_i[k] - M_line_j[k]);
					}
					if (dist > max_dist) {
						max_dist = dist;
					}
				}
			}
		}
	}

	return(max_dist);
}

/******************************************************************************/

void find_all_leaves(Node *node, Node **cluster_nodes, unsigned long *cluster_num)
{
	if (node != NULL) {
		find_all_leaves(node->leftchild,  cluster_nodes, cluster_num);
		find_all_leaves(node->rightchild, cluster_nodes, cluster_num);
		/*
		 * The PDDP algorithm ensures that every node has either 2 children or is a leaf node.
		 * Hence, a leaf node is found when both pointers to the children are equal (both are NULL).
		 */
		if (((node->keep_in_tree == 1) || (node->keep_in_tree == 2)) && ((node->leftchild == node->rightchild) || (node->leftchild->keep_in_tree == 0) || (node->rightchild->keep_in_tree == 0))) {
			cluster_nodes[*cluster_num] = node;
			(*cluster_num)++;
		}
	}
}

/******************************************************************************/

__attribute__ ((target(mic))) void find_all_splittable_leaves(Node *node, Node **cluster_nodes, unsigned long *cluster_num)
{
	if (node != NULL) {
		find_all_splittable_leaves(node->leftchild,  cluster_nodes, cluster_num);
		find_all_splittable_leaves(node->rightchild, cluster_nodes, cluster_num);
		/*
		 * The PDDP algorithm ensures that every node has either 2 children or is a leaf node.
		 * Hence, a leaf node is found when both pointers to the children are equal (both are NULL).
		 */
		if ((node->leftchild == node->rightchild) && (node->is_splittable == 1)) {
			cluster_nodes[*cluster_num] = node;
			(*cluster_num)++;
		}
	}
}

/******************************************************************************/

double min_cluster_distance(Node *node, unsigned long clusters, unsigned long attributes)
{
	Node		**cluster_nodes;
	unsigned long	i, j, k, cluster_num = 0;
	int		error;
	double		dist, min_dist = INFINITY;

	error = posix_memalign((void **)(&cluster_nodes), 64, clusters * sizeof(Node *));

	if (error != 0) {
		printf("ERROR: Could not allocate memory for vector of cluster nodes.\n");
		exit(0);
	}

	find_all_leaves(node, cluster_nodes, &cluster_num);
	
	printf("Found %ld leaf nodes to keep\n", cluster_num);

	for (i = 0; i < clusters; i++) {
		for (j = i + 1; j < clusters; j++) {
			dist = 0.0;
			for (k = 0; k < attributes; k++) {
				dist += (cluster_nodes[i]->centroid[k] - cluster_nodes[j]->centroid[k]) * (cluster_nodes[i]->centroid[k] - cluster_nodes[j]->centroid[k]);
			}
			if (dist < min_dist) {
				min_dist = dist;
			}
		}
	}

	free(cluster_nodes);

	return(min_dist);
}

/******************************************************************************/

void post_order_traversal(Node *node, unsigned long *indices, unsigned long *cluster)
{
	if (node != NULL) {
		post_order_traversal(node->leftchild,  indices, cluster);
		post_order_traversal(node->rightchild, indices, cluster);
		/*
		 * The PDDP algorithm ensures that every node has either 2 children or is a leaf node.
		 * Hence, a leaf node is found when both pointers to the children are equal (both are NULL).
		 */
		if (((node->keep_in_tree == 1) || (node->keep_in_tree == 2)) && ((node->leftchild == node->rightchild) || (node->leftchild->keep_in_tree == 0) || (node->rightchild->keep_in_tree == 0))) {
			unsigned long	i;

			for (i = 0; i < node->num_of_indices; i++) {
				indices[node->indices[i]] = *cluster;
			}
			(*cluster)++;
		}
	}
}

/******************************************************************************/

void assign_cluster_numbers(Node *node, unsigned long *result, unsigned long data_points)
{
	unsigned long	cluster = 1;

	post_order_traversal(node, result, &cluster);
}

/******************************************************************************/

#pragma offload_attribute (pop)

/******************************************************************************/

void write_results(unsigned long *result, unsigned long data_points, char *filename)
{
	unsigned long	i;
	FILE		*file;

	file = fopen(filename, "w");

	if (file == NULL) {
		printf("ERROR: Could not open output file.\n");
		exit(0);
	}

	fprintf(file, "[");

	for (i = 0; i < data_points - 1; i++) {
		fprintf(file, "%ld, ", result[i]);
	}

	fprintf(file, "%ld]", result[data_points - 1]);

	fclose(file);
}

/******************************************************************************/

int main(int argc, char *argv[])
{
	unsigned long	i, j, data_points, attributes, padded_attributes, clusters, *result;
	int		error;
	FILE		*file;
	char		buffer[BUFFER_SIZE], *line = buffer, *token, *endptr;
	double		*M;
	struct timeval	start, end;

	__assume_aligned(M, 64);

	if (argc < 4) {
		printf("ERROR: Insufficient number of arguments.\n");
		exit(0);
	}

	clusters = strtol(argv[3], &endptr, 10);
	if (*endptr != '\0' || clusters < 2) {
		printf("ERROR: Invalid number of clusters to build.\n");
		exit(0);
	}

	gettimeofday(&start, NULL);

	file = fopen(argv[1], "r");

	if (file == NULL) {
		printf("ERROR: Could not open input file.\n");
		exit(0);
	}

	/*
	 * Count how many lines the input data file contains.
	 */
	data_points = 0;
	while (fgets(buffer, BUFFER_SIZE, file) != NULL) {
		data_points++;
	}

	printf("Lines in input file: %ld\n", data_points);

	if (data_points < 2) {
		printf("ERROR: The number of data points to be clustered must be greater than 1.\n");
		exit(0);
	}

	active_data_points = data_points;

	/*
 	 * Allocate memory to store output data.
 	 * We use mmap() since it is faster than malloc() and aligns the starting address on a page boundary.
 	 */
	result = (unsigned long *)mmap(NULL, data_points * sizeof(unsigned long), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

	if (result == MAP_FAILED) {
		printf("ERROR: Could not allocate memory for output vector.\n");
		exit(0);
	}

	/*
	 * Count how many attributes a line consists of.
	 * Attributes in the input data file are seperated by a comma (','), except the last attribute.
	 * We assume that each line contains at least one attribute.
	 */
	attributes = 1;
	for (; *line; attributes += *line == ',', line++);

	printf("Attributes in each line: %ld\n", attributes);

	/*
	 * Pad each line to make its size a multiple of 64 bytes.
	 * Useful for performance reasons, as it allows better vectorization on the Intel Xeon Phi.
	 */
	padded_attributes = (((attributes * sizeof(double) - 1) | (ALIGNMENT - 1)) + 1) / sizeof(double);

	printf("Padded attributes in each line: %ld\n", padded_attributes);

	/*
	 * Allocate memory to store input data.
	 * We use mmap() since it is faster than malloc() and aligns the starting address on a page boundary.
	 */
	M = (double *)mmap(NULL, data_points * padded_attributes * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

	if (M == MAP_FAILED) {
		printf("ERROR: Could not allocate memory for input array.\n");
		exit(0);
	}

	/*
	 * Return to the beginning of the file to start reading the input data.
	 */
	fseek(file, 0, SEEK_SET);

	i = 0;
	j = 0;
	while((line = fgets(buffer, BUFFER_SIZE, file)) != NULL) {
		token = strtok(line, ",");
		while (token != NULL) {
			M[i * padded_attributes + j] = atof(token);
			token = strtok(NULL, ",");
			j++;
			if (j == attributes) {
				j = 0;
				i++;
			}
		}
	}

	fclose(file);

	gettimeofday(&end, NULL);

	print_elapsed_time(start, end, "Time to read input data");

#pragma offload target(mic)	in(leaves, data_points, active_data_points, attributes, padded_attributes) \
				in(M : length(data_points * padded_attributes) align(64)) \
				out(result : length(data_points) align(64)) \
				inout(clusters)\
				nocopy(num_of_cores)
{
	__attribute__((aligned(64))) unsigned long	i, j, current_level, max_level, stop, cluster_num, *indices;
	__attribute__((aligned(64))) int		error;
	__attribute__((aligned(64))) double		*M_line, *centroid;
	__attribute__((aligned(64))) Node		*root, **cluster_nodes;
	__attribute__((aligned(64))) double		max_diam, min_dist;
	__attribute__((aligned(64))) struct timeval	start, end;

	__assume_aligned(root, 64);
	__assume_aligned(M_line, 64);
	__assume_aligned(indices, 64);
	__assume_aligned(centroid, 64);
	__assume_aligned(cluster_nodes, 64);

	#if defined(_OPENMP)
	#pragma omp parallel
	{
	#pragma omp single
	{
	num_of_cores = omp_get_num_threads();
	}
	}
	#else
	num_of_cores = 1;
	#endif

	printf("Detected %lu processors.\n", num_of_cores);

	gettimeofday(&start, NULL);

	/*
	 * Make an initial guess about the depth of the tree to be constructed.
	 * We will construct a perfect binary tree in which the number of leaves is greater or equal to the number of requested clusters.
	 */
	i         = clusters;
	max_level = -1;

	while (i > 0) {
		i >>= 1;
		max_level++;
	}

	if ((clusters & (clusters - 1))) {
		max_level++;
	}

	/*
	 * Allocate the root node of the tree.
	 */
	root = allocate_node(data_points);
	DBG("Root is %p\n", root);

	indices  = (unsigned long *)mmap(NULL, data_points * sizeof(unsigned long), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	centroid = (double *)mmap(NULL, attributes  * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

	if ((indices == MAP_FAILED) || (centroid == MAP_FAILED)) {
		printf("ERROR: Could not allocate memory for a vector in tree node.\n");
		exit(0);
	}

	/*
	 * All data points belong to the root node at the beginning.
	 */
	#pragma vector aligned
	#pragma ivdep
	for (i = 0; i < data_points; i++) {
		indices[i] = i;
	}

	/*
	 * Initialize centroid.
	 */
	memset(centroid, 0, attributes * sizeof(double));

	#pragma omp parallel default(none) private(i, j, M_line) shared(M, indices, centroid, data_points, attributes, padded_attributes)
	{
	__attribute__((aligned(64))) double	centroid_local[attributes];

	memset(centroid_local, 0, attributes * sizeof(double));

	#pragma omp for
	for (i = 0; i < data_points; i++) {
		M_line = &M[indices[i] * padded_attributes];
		#pragma vector aligned
		#pragma ivdep
		for (j = 0; j < attributes; j++) {
			centroid_local[j] += M_line[j];
		}
	}

	#pragma vector aligned
	#pragma ivdep
	for (j = 0; j < attributes; j++) {
		#pragma omp atomic
		centroid[j] += centroid_local[j];
	}
	}

	#pragma vector aligned
	#pragma ivdep
	for (i = 0; i < attributes; i++) {
		centroid[i] /= data_points;
	}

	init_node(root, root, M, data_points, indices, centroid, attributes, padded_attributes, num_of_cores);

	current_level = 0;
	stop = 0;

#pragma omp parallel private(i) num_threads(num_of_cores < ((1 << max_level) / 4) ? num_of_cores : ((1 << max_level) / 4))
	{
#pragma omp single
	{
	do {
		cluster_num = 0;
#pragma omp taskgroup
		{
		error = posix_memalign((void **)(&cluster_nodes), 64, (1 << current_level) * sizeof(Node *));

		if (error != 0) {
			printf("ERROR: Could not allocate memory for vector of cluster nodes.\n");
			exit(0);
		}

		find_all_splittable_leaves(root, cluster_nodes, &cluster_num);

		for (i = 0; i < cluster_num; i++) {
			#pragma omp task
			process_node(cluster_nodes[i], M, attributes, padded_attributes, current_level, max_level);
		}
		}	// #pragma omp taskgroup

		free(cluster_nodes);

		/*
		 * If there have not been created as many clusters as requested,
		 * then we have to continue creating new clusters.
		 * 
		 * Except if there are no more clusters that can be split.
		 */
		if (leaves < clusters) {
			if (cluster_num == 0) {
				stop = 1;
				clusters = leaves;
				keep_all_nodes(root);
			}
		} else {
			/*
			 * The number of clusters created is equal or larger to the number of clusters requested.
			 * Check which ones have to be kept and if they are the ones required for the proper solution.
			 */
			stop = find_nodes_to_keep(root, clusters);
		}

		current_level = max_level;
		max_level++;
	} while (stop == 0);
	}	// #pragma omp single
	}	// #pragma omp parallel

	printf("Created a total of %ld clusters.\n", clusters);

	gettimeofday(&end, NULL);

	print_elapsed_time(start, end, "Time for calculations");

	gettimeofday(&start, NULL);

	assign_cluster_numbers(root, result, data_points);

	gettimeofday(&end, NULL);

	print_elapsed_time(start, end, "Time to assign cluster numbers");

	gettimeofday(&start, NULL);

	max_diam = sqrt(max_cluster_diameter(root, M, attributes, padded_attributes));

	gettimeofday(&end, NULL);

	print_elapsed_time(start, end, "Time to calculate maximum cluster diameter");

	printf("Maximum cluster diameter is %f\n", max_diam);

	gettimeofday(&start, NULL);

	min_dist = sqrt(min_cluster_distance(root, clusters, attributes));

	gettimeofday(&end, NULL);

	print_elapsed_time(start, end, "Time to calculate minimum cluster distance");

	printf("Minimum cluster distance is %f\n", min_dist);

	printf("Dunn index = %.40f\n", min_dist / max_diam);
}	// #pragma offload

	gettimeofday(&start, NULL);

        write_results(result, data_points, argv[2]);

        gettimeofday(&end, NULL);

        print_elapsed_time(start, end, "Time to write output data");

	return(0);
}

