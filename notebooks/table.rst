.. list-table::
   :header-rows: 1
   :widths: 14 15 19 25 20

   * - Method name
     - Parameters
     - Scalability
     - Usecase
     - Geometry (metric used)

   * - `K-Means <k_means>`
     - number of clusters
     - Very large ``n_samples``, medium ``n_clusters`` with
       
     - General-purpose, even cluster size, flat geometry, not too many clusters
     - Distances between points

   * - `Affinity propagation <affinity_propagation>`
     - damping, sample preference
     - Not scalable with n_samples
     - Many clusters, uneven cluster size, non-flat geometry
     - Graph distance (e.g. nearest-neighbor graph)

   * - `Mean-shift <mean_shift>`
     - bandwidth
     - Not scalable with ``n_samples``
     - Many clusters, uneven cluster size, non-flat geometry
     - Distances between points

   * - `Spectral clustering <spectral_clustering>`
     - number of clusters
     - Medium ``n_samples``, small ``n_clusters``
     - Few clusters, even cluster size, non-flat geometry
     - Graph distance (e.g. nearest-neighbor graph)

   * - `Ward hierarchical clustering <hierarchical_clustering>`
     - number of clusters
     - Large ``n_samples`` and ``n_clusters``
     - Many clusters, possibly connectivity constraints
     - Distances between points

   * - `Agglomerative clustering <hierarchical_clustering>`
     - number of clusters, linkage type, distance
     - Large ``n_samples`` and ``n_clusters``
     - Many clusters, possibly connectivity constraints, non Euclidean
       distances
     - Any pairwise distance

   * - `DBSCAN <dbscan>`
     - neighborhood size
     - Very large ``n_samples``, medium ``n_clusters``
     - Non-flat geometry, uneven cluster sizes
     - Distances between nearest points

   * - `Gaussian mixtures <mixture>`
     - many
     - Not scalable
     - Flat geometry, good for density estimation
     - Mahalanobis distances to  centers

   * - `Birch`
     - branching factor, threshold, optional global clusterer.
     - Large ``n_clusters`` and ``n_samples``
     - Large dataset, outlier removal, data reduction.
     - Euclidean distance between points
