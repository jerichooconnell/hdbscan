#print(__doc__)
from __future__ import print_function
from past.utils import old_div

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.cm as cm
import hdbscan
import skfuzzy as fuzz

from sklearn import metrics
from sklearn.decomposition import PCA

from sklearn.metrics import homogeneity_score,homogeneity_completeness_v_measure
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.decomposition import NMF, FastICA, PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from itertools import cycle, islice
from scipy.io import loadmat
from scipy import mod
from shogun import Jade, RealFeatures

def plot_results(sc=0,sr=0,sv=1,sv2=0,srs=0,mode=[],w_positions=False,scale=False,algo_params=[],rng=10,cinds=[0,2,3,4,7,8,9,10],return_vs=0,return_bars=0,dinds=[0,1,2,3,4],save=False,title='misc',red=[], **kwargs
):
    
    '''
    sc: bool
    show clusters plot
    sr: bool
    show reconstruction plot
    sv: bool
    show vscores
    mode: string
    [] = pca, 'all'= all, 'de' = dual energy,'se' = integrating detector 
    W_position: bool
    Add positions to the vectors
    scale: bool
    Rescale the input, this is important if you add the position data
    default_base: dict
    Parameters for the clustering algorithms
    rng: int
    seed
    cinds: list
    which clustering methods to try
    red: string
    Which demensional REDuction 'ica', 'nmf' or 'tsne'
    '''

    np.random.seed(rng)


    # ============
    # Set up cluster parameters
    # ============
    if sc:
        plt.figure(1,figsize=(9 * 2 + 3, 12.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)
    if sr:
        plt.figure(2,figsize=(9 * 2 + 3, 12.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)
    if srs:
        plt.figure(3,figsize=(9 * 2 + 3, 12.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)


    plot_num = 1


    # This dictionary defines the colormap
    cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0
                      (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1

            'green': ((0.0, 0.8, 0.8),   # set to 0.8 so its not too bright at 0
                      (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.0, 0.0)),  # no green at 1

            'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
                      (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.0, 0.0))   # no blue at 1
           }

    # Create the colormap using the dictionary
    P = color.LinearSegmentedColormap('GnRd', cdict)

    homo, comp, vs, idata, ialgo = [],[],[],[],[]
    
    data = (('glass','Glass'),('pp','Poly'),('bb','Bluebelt'),('ptfe','PTFE'),('steel','Steel'
            ))
    
    datasets2 = [data[j] for j in dinds]

    default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -60,
                'n_neighbors': 2,
                'n_clusters': 2,
                'linkage': 'ward',
                'affinity': "nearest_neighbors",
                'assign_labels':'kmeans',
                'min_samples':10,
                'ct':'spherical',
                'branching':19,
                'threshold':0.0001,
                'metric':'minkowski',
                'asc':False,
                'p':2,
                'mcs':10,
                'nc':2}
    
    for i_dataset, (dataset,dat_name) in enumerate(datasets2):

        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)


        if mode == 'de':
            X = loadmat('all'+dataset)['Z'][:,1:]
            X = np.column_stack((np.sum(X[:,1:3],1),np.sum(X[:,4:],1)))
        elif mode == 'se':
            X = np.reshape(X, (20*68,1), order="F") 
        elif mode == 'all':
            X = loadmat('all'+dataset)['Z'][:,1:]
        elif mode == 'small':
            X = loadmat('small_'+dataset)['Z']            
        else:
            X = loadmat('2'+dataset)['Z']

        label_true = loadmat('2'+dataset+'_mask')['BW']

        if w_positions:
            xx,yy = np.meshgrid(range(68),range(20))        
            x = np.reshape(xx, (20*68,1), order="F")
            y = np.reshape(yy, (20*68,1), order="F")
            X = np.concatenate((X,x,y),axis = 1)

        if scale:
            X = StandardScaler().fit_transform(X)
            
        if red == 'ica':
            X = FastICA(n_components=params['nc']).fit_transform(X)
        if red == 'tsne':
            X = TSNE(n_components=params['nc']).fit_transform(X)
        if red == 'nmf':
            X = NMF(n_components=params['nc']).fit_transform(X) 
            
        if red == 'pca':
            X = PCA(n_components=params['nc']).fit_transform(X)  
        if red == 'spec':
            X = SpectralEmbedding(n_components=3).fit_transform(X)

            
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
 
        # ============
        # Create cluster objects
        # ============
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        ward = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage=params['linkage'],
            connectivity=connectivity)
        spectral = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity=params['affinity'],assign_labels=params['assign_labels'])
        dbscan = cluster.DBSCAN(eps=params['eps'],min_samples=params['min_samples'],metric=params['metric'],p=params['p'])
        affinity_propagation = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'])
        average_linkage = cluster.AgglomerativeClustering(
            linkage="complete", affinity="cityblock",
            n_clusters=params['n_clusters'], connectivity=connectivity)
        birch = cluster.Birch(n_clusters=params['n_clusters'],branching_factor=params['branching'],
                             threshold=params['threshold'])
        gmm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type=params['ct'])
        bgmm = mixture.BayesianGaussianMixture(
            n_components=params['n_clusters'], covariance_type=params['ct'])
        hdb = hdbscan.HDBSCAN(min_samples=params['min_samples'],min_cluster_size=params['mcs'],metric=params['metric'],allow_single_cluster=params['asc'],p=params['p'], **kwargs)
        
        cinds_all = (
            ('KMeans', two_means),
            ('AffinityPropagation', affinity_propagation),
            ('MeanShift', ms),
            ('SpectralClustering', spectral),
            ('Ward', ward),
            ('AgglomerativeClustering', average_linkage),
            ('DBSCAN', dbscan),
            ('Birch', birch),
            ('HDBSCAN', hdb),
            ('GaussianMixture', gmm),
            ('BGaussianMixture', bgmm)
        )

        clustering_algorithms = [cinds_all[j] for j in cinds]

        for i_algorithm, (name, algorithm) in enumerate(clustering_algorithms):

            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                if hasattr(algorithm, 'condensed_tree_'):
                    pass
                else:
                    algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            elif hasattr(algorithm, 'condensed_tree_'):
                y_pred = algorithm.fit_predict(X)
            else:
                y_pred = algorithm.predict(X)
                
            homo1,comp1,vs1 = homogeneity_completeness_v_measure(label_true.squeeze(), y_pred)
            
            if sc:
                plt.figure(1)

                plt.rcParams['axes.facecolor'] = P(1 - vs1,alpha=0.5)

                plt.subplot(len(datasets2), len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)
                if i_algorithm == 0:
                    plt.ylabel(dat_name, size=18)


                colors2 = np.array(list(islice(cycle(['r','b', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']),
                                              int(max(y_pred) + 1))))

                # add black color for outliers (if any)
                colors2 = np.append(colors2, ["#000000"])
                #import ipdb; ipdb.set_trace()
                X2 = X[label_true.squeeze() == 0,:]
                plt.scatter(X2[:, 0], X2[:,1], s=40, color=colors2[y_pred[label_true.squeeze() == 0]])
                
                X2 = X[label_true.squeeze() == 1,:]
                plt.scatter(X2[:, 0], X2[:,1], s=40, color=colors2[y_pred[label_true.squeeze() == 1]],marker='x')

                plt.xticks(())
                plt.yticks(())
                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='right')
                plt.text(.99, .89, ('%.2f' % vs1).lstrip('0'),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='right')
#             if sr | srs:
#                 aa = y_pred[1320]
                
#                 if aa != 0:
#                     y_pred[y_pred == y_pred.min()] = y_pred.max() + 1                    
#                     y_pred[y_pred == aa] = y_pred.min()
            if sr:        
                plt.figure(2)
                r = np.reshape(y_pred, (20,68), order="F")
                plt.subplot(len(datasets2), len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)
                plt.imshow(r)
                plt.set_cmap('bwr')
                plt.xticks([])
                plt.yticks([])        
                if i_algorithm == 0:
                    plt.ylabel(dat_name, size=18)
            if srs:
                plt.figure(3)
                r = np.reshape(y_pred, (20,68), order="F")
                p = np.reshape(label_true, (20,68), order="F")
                plt.subplot(len(datasets2), len(clustering_algorithms), plot_num)
                # Bin all of the bins to one and zero
                
                r[r == r.min()] = 0
                r[r > r.min()] = 1
                
                if i_dataset == 0:
                    plt.title(name, size=18)
                plt.imshow(abs(p - r))
                plt.set_cmap('gray')
                plt.xticks([])
                plt.yticks([])        
                if i_algorithm == 0:
                    plt.ylabel(dat_name, size=18)
                    
            homo.append(homo1)
            comp.append(comp1)
            vs.append(vs1)

            idata.append(i_dataset)
            ialgo.append(i_algorithm)

            plot_num += 1
    
    if save:
        plt.figure(1)
        plt.savefig('scatter_'+title+'.png')
        plt.figure(2)
        plt.savefig('recon_'+title+'.png')
    
    plt.show()
        
    if sv:
        plt.rcParams['axes.facecolor'] = (0,0,0)    
        vs = np.asarray(vs)
        bars = []
        n_components_range = range(len(clustering_algorithms))
        cv_types = [item[1] for item in datasets2]


        color_iter = cycle(['navy', 'turquoise', 'cornflowerblue',
                                      'darkorange','k'])
        # Plot the BIC scores
        plt.figure(figsize=(8, 6))
        spl = plt.subplot(1, 1, 1)

        for i, (cv_type, pcolor) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + .166 * (i - 2)
            bars.append(plt.bar(xpos, vs[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)],
                                width=.166, color=pcolor))
        plt.xticks(n_components_range,[item[0] for item in clustering_algorithms])
        plt.xticks()
        plt.ylim([0, 1])
        plt.title('V score per model')
        xpos = np.mod(vs.argmax(), len(n_components_range)) + .65 +\
            .16 * np.floor(vs.argmax() / len(n_components_range))
        plt.text(xpos, vs.min() * 0.97 + .03 * vs.max(), '*', fontsize=14)
        spl.set_xlabel('Algorithm')
        spl.legend([b[0] for b in bars], cv_types)  
        plt.tight_layout()
        
    if sv2:
        
        vs = np.asarray(vs)
        bars = []
        colors = []
        n_components_range = range(len(clustering_algorithms))
        cv_types = [item[1] for item in datasets2]


        color_iter = cycle(['navy', 'turquoise', 'cornflowerblue',
                                      'darkorange','k'])
        # Plot the BIC scores
        plt.figure(figsize=(8, 6))
        plt.rcParams['axes.facecolor'] = (1,1,1)            
        spl = plt.subplot(1, 1, 1)

        for i, (cv_type, pcolor) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + 0.166 * (i - 2)
            bars.append( vs[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)])
        
        # import ipdb; ipdb.set_trace()
        [colors.append(colourblind(col)) for col in n_components_range]
        
        bars2 = np.mean(np.asarray(bars),axis=0)
        
        indeces = np.argsort(bars2)
#         import ipdb; ipdb.set_trace()
        bars2.sort()
        clustering_algorithms = [clustering_algorithms[i] for i in indeces]
        
        plt.bar(n_components_range,bars2,color=colors)
        plt.xticks(n_components_range,[item[0] for item in clustering_algorithms],rotation = 45, ha="right")
        plt.xticks()
        plt.ylim([0, 1])
        plt.title('V Averaged over all Materials')
        xpos = np.mod(vs.argmax(), len(n_components_range)) + .65 +\
            .16 * np.floor(vs.argmax() / len(n_components_range))
        plt.text(xpos, vs.min() * 0.97 + .03 * vs.max(), '*', fontsize=14)
        spl.set_xlabel('Algorithm')
        spl.set_ylabel('V-score')
        #spl.legend([b[0] for b in bars], cv_types)  
        plt.tight_layout()
        
        if return_bars:
            return bars
    if return_vs:    
        return vs

def silhouette_analysis(dataset='glass',mode=[],scale=False,w_positions=False,plotting=True,include_1=False):

    # Generating the sample data from make_blobs
    # This particular setting has one distinct cluster and 3 clusters placed close
    # together.
    
    if mode == 'de':
        X = loadmat('all'+dataset)['Z'][:,1:]
        X = np.column_stack((np.sum(X[:,1:3],1),np.sum(X[:,4:],1)))
        
    elif mode == 'se':
        X = loadmat('all'+dataset)['Z'][:,0]
        X = np.reshape(X, (20*68,1), order="F")
        
    elif mode == 'all':
        X = loadmat('all'+dataset)['Z'][:,1:]
    else:
        X = loadmat('2'+dataset)['Z']

    if w_positions:
        xx,yy = np.meshgrid(range(68),range(20))        
        x = np.reshape(xx, (20*68,1), order="F")
        y = np.reshape(yy, (20*68,1), order="F")
        X = np.concatenate((X,x,y),axis = 1)

    if scale:
        X = StandardScaler().fit_transform(X) 
    
    if include_1:
        range_n_clusters = [1, 2, 3, 4, 5, 6]
    else:
        range_n_clusters = [2, 3, 4, 5, 6]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        if plotting:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        
        if plotting:
            
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

        plt.show()
        
def kmeans_finetune(dataset='glass',mode=[],scale=False,w_positions=False,plotting=True,include_1=False):
    
    np.random.seed(42)
    # Generating the sample data from make_blobs
    # This particular setting has one distinct cluster and 3 clusters placed close
    # together.
    
    if mode == 'de':
        X = loadmat('all'+dataset)['Z'][:,1:]
        X = np.column_stack((np.sum(X[:,1:3],1),np.sum(X[:,4:],1)))
        
    elif mode == 'se':
        X = loadmat('all'+dataset)['Z'][:,0]
        X = np.reshape(X, (20*68,1), order="F")
        
    elif mode == 'all':
        X = loadmat('all'+dataset)['Z'][:,1:]
    else:
        X = loadmat('2'+dataset)['Z']

    if w_positions:
        xx,yy = np.meshgrid(range(68),range(20))        
        x = np.reshape(xx, (20*68,1), order="F")
        y = np.reshape(yy, (20*68,1), order="F")
        X = np.concatenate((X,x,y),axis = 1)

    if scale:
        X = StandardScaler().fit_transform(X) 
    
    data = X
    
    n_samples, n_features = data.shape
    n_digits = 2
    
    labels = loadmat('2'+dataset+'_mask')['BW'].squeeze()

    sample_size = 300

    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_digits, n_samples, n_features))


    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


    def bench_k_means(estimator, name, data):
        t0 = time.time()
        estimator.fit(data)
        print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
              % (name, (time.time() - t0), estimator.inertia_,
                 metrics.homogeneity_score(labels, estimator.labels_),
                 metrics.completeness_score(labels, estimator.labels_),
                 metrics.v_measure_score(labels, estimator.labels_),
                 metrics.adjusted_rand_score(labels, estimator.labels_),
                 metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=sample_size)))

    bench_k_means(cluster.KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                  name="k-means++", data=data)

    bench_k_means(cluster.KMeans(init='random', n_clusters=n_digits, n_init=10),
                  name="random", data=data)

    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=n_digits).fit(data)
    bench_k_means(cluster.KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
                  name="PCA-based",
                  data=data)
    print(82 * '_')

    # #############################################################################
    # Visualize the results on PCA-reduced data
    if plotting:
        reduced_data = PCA(n_components=2).fit_transform(data)
        kmeans = cluster.KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                  'Centroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        
# def plot_fuzzy(dataset='glass'):

#     colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
    
#     X = loadmat('2'+dataset)['Z'].transpose()

   
#     # Set up the loop and plot
#     plt.figure()
    
#     fpcs = []

#     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#         X, 2, 2, error=0.005, maxiter=1000, init=None)

#     # Store fpc values for later
#     fpcs.append(fpc)

#     # Plot assigned clusters, for each data point in training set
#     cluster_membership = np.argmax(u, axis=0)
    
#     for j in range(2):
#         plt.plot(X[0,cluster_membership == j],
#                 X[1,cluster_membership == j], '.', color=colors[j])

#     # Mark the center of each fuzzy cluster
#     for pt in cntr:
#         plt.plot(pt[0], pt[1], 'rs')

#     plt.title('Centers = {0}; FPC = {1:.2f}'.format(2, fpc))
#     plt.axis('off')

#     plt.show()
    
#     label_true = loadmat('2'+dataset+'_mask')['BW']

    
#     homo1,comp1,vs1 = homogeneity_completeness_v_measure(label_true.squeeze(), cluster_membership)
    
    print('V score',vs1,'fpc',fpc)
    
def plot_fuzzy(sc=0,sr=0,sv=1,mode=[],w_positions=False,scale=False,algo_params=[],rng=10,cinds=[0,6,8,9],m=2
):
    
    '''
    sc: bool
    show clusters plot
    sr: bool
    show reconstruction plot
    sv: bool
    show vscores
    mode: string
    [] = pca, 'all'= all, 'de' = dual energy,'se' = integrating detector 
    W_position: bool
    Add positions to the vectors
    scale: bool
    Rescale the input, this is important if you add the position data
    default_base: dict
    Parameters for the clustering algorithms
    rng: int
    seed
    cinds: list
    which clustering methods to try
    '''

    np.random.seed(rng)

    i_algorithm = 0
    clustering_algorithms = (('Fuzzy cmeans',i_algorithm))
    # ============
    # Set up cluster parameters
    # ============
    if sc:
        plt.figure(1,figsize=(9 * 2 + 3, 12.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)
    if sr:
        plt.figure(2,figsize=(9 * 2 + 3, 12.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)


    plot_num = 1


    # This dictionary defines the colormap
    cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0
                      (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1

            'green': ((0.0, 0.8, 0.8),   # set to 0.8 so its not too bright at 0
                      (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.0, 0.0)),  # no green at 1

            'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
                      (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.0, 0.0))   # no blue at 1
           }

    # Create the colormap using the dictionary
    P = color.LinearSegmentedColormap('GnRd', cdict)

    homo, comp, vs, idata, ialgo = [],[],[],[],[]
    
    datasets2 = (('glass','Glass'),('pp','Poly'),('bb','Bluebelt'),('ptfe','PTFE'),('steel','Steel'
            ))

    default_base = {'quantile': .3,
                'eps': .5,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 2,
                'n_clusters': 2}
    
    for i_dataset, (dataset,dat_name) in enumerate(datasets2):

        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)


        if mode == 'de':
            X = loadmat('all'+dataset)['Z'][:,1:]
            X = np.column_stack((np.sum(X[:,1:3],1),np.sum(X[:,4:],1)))
        elif mode == 'se':
            X = loadmat('all'+dataset)['Z'][:,0]
            X = np.reshape(X, (20*68,1), order="F") 
        elif mode == 'all':
            X = loadmat('all'+dataset)['Z'][:,1:]
        else:
            X = loadmat('2'+dataset)['Z']

        label_true = loadmat('2'+dataset+'_mask')['BW']

        if w_positions:
            xx,yy = np.meshgrid(range(68),range(20))        
            x = np.reshape(xx, (20*68,1), order="F")
            y = np.reshape(yy, (20*68,1), order="F")
            X = np.concatenate((X,x,y),axis = 1)

        if scale:
            X = StandardScaler().fit_transform(X)    


        
        t0 = time.time()
        
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.transpose(), 2, m, error=0.005, maxiter=1000, init=None)

        # Plot assigned clusters, for each data point in training set
        y_pred = np.argmax(u, axis=0)

        t1 = time.time()
        
        homo1,comp1,vs1 = homogeneity_completeness_v_measure(label_true.squeeze(), y_pred)

        if sc:
            plt.figure(1)

            plt.rcParams['axes.facecolor'] = P(1 - vs1,alpha=0.5)

            plt.subplot(len(datasets2), 1, plot_num)
            if i_dataset == 0:
                plt.title('fuzzy cmeans', size=18)
            if i_algorithm == 0:
                plt.ylabel(dat_name, size=18)


            colors2 = np.array(list(islice(cycle(['r','b', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))

            # add black color for outliers (if any)
            colors2 = np.append(colors2, ["#000000"])
            plt.scatter(X[:, 0], X[:,1], s=10, color=colors2[y_pred])

            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plt.text(.99, .89, ('%.2f' % vs1).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
        if sr:
            plt.figure(2)
            r = np.reshape(y_pred, (20,68), order="F")
            plt.subplot(len(datasets2), 1, plot_num)
            if i_dataset == 0:
                plt.title('fuzzy cmeans', size=18)
            plt.imshow(r)
            plt.xticks([])
            plt.yticks([])        
            if i_algorithm == 0:
                plt.ylabel(dat_name, size=18) 

        homo.append(homo1)
        comp.append(comp1)
        vs.append(vs1)

        idata.append(i_dataset)
        ialgo.append(i_algorithm)

        plot_num += 1

    plt.show()
    
    if sv:
        
        vs = np.asarray(vs)
        bars = []
        n_components_range = [1]
        cv_types = [item[1] for item in datasets2]


        color_iter = cycle(['navy', 'turquoise', 'cornflowerblue',
                                      'darkorange','k'])
        # Plot the BIC scores
        plt.figure(figsize=(8, 6))
        spl = plt.subplot(1, 1, 1)

        for i, (cv_type, pcolor) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + .166 * (i - 2)
            bars.append(plt.bar(xpos, vs[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)],
                                width=.166, color=pcolor))
        plt.xticks(n_components_range,['Fuzzy cmeans'])
        plt.xticks()
        plt.ylim([0, 1])
        plt.title('V score per model')
        xpos = np.mod(vs.argmax(), len(n_components_range)) + .65 +\
            .16 * np.floor(vs.argmax() / len(n_components_range))
        plt.text(xpos, vs.min() * 0.97 + .03 * vs.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], cv_types)  
    
def compare_outputs(vs1,vs2):
    
    vs = np.asarray(vs1)
    
    datasets2 = (('glass','Glass'),('pp','Poly'),('bb','Bluebelt'),('ptfe','PTFE'),('steel','Steel'
            ))
    
    bars = []
    n_components_range = [1]
    cv_types = [item[1] for item in datasets2]


    color_iter = cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange','k'])
    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(1, 1, 1)

    for i, (cv_type, pcolor) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .16 * (i - 2)
        bars.append(plt.bar(xpos, vs[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.08, color=pcolor))
    plt.xticks(n_components_range,['Fuzzy cmeans'])
    plt.xticks()
    plt.ylim([0, 1])
    plt.title('V score per model')
    xpos = np.mod(vs.argmax(), len(n_components_range)) + .65 +\
        .16 * np.floor(vs.argmax() / len(n_components_range))
    plt.text(xpos, vs.min() * 0.97 + .03 * vs.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    
    vs = np.asarray(vs2)
    
    bars = []
    n_components_range = [1]
    cv_types = [item[1] for item in datasets2]


    color_iter = cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange','k'])

    for i, (cv_type, pcolor) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .16 * (i - 1.5)
        bars.append(plt.bar(xpos, vs[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.08, color=pcolor))
    plt.xticks(n_components_range,['Fuzzy cmeans'])
    plt.xticks()
    plt.ylim([0, 1])
    plt.title('V score per model')
    xpos = np.mod(vs.argmax(), len(n_components_range)) + .65 +\
        .16 * np.floor(vs.argmax() / len(n_components_range))
    plt.text(xpos, vs.min() * 0.97 + .03 * vs.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    
def colourblind(i):
    '''
        colour pallete from http://tableaufriction.blogspot.ro/
        allegedly suitable for colour-blind folk
        SJ
    '''

    rawRGBs = [(162,200,236),
               (255,128,14),
               (171,171,171),
               (95,158,209),
               (89,89,89),
               (0,107,164),
               (255,188,121),
               (207,207,207),
               (200,82,0),
               (137,137,137)]

    scaledRGBs = []
    for r in rawRGBs:
        scaledRGBs.append((old_div(r[0],255.),old_div(r[1],255.),old_div(r[2],255.)))

    idx = mod(i,len(scaledRGBs))
    return scaledRGBs[idx]

def colourblind2(i):
    '''
        another colour pallete from http://www.sron.nl/~pault/
        allegedly suitable for colour-blind folk
        SJ
    '''

    hexcols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77',
               '#CC6677', '#882255', '#AA4499']
    idx = mod(i,len(hexcols))
    return hexcols[idx]

def compare_3(bars_se,bars_de,bars_pc,title='all',leg=['Single Energy','Dual Energy','Spectral']):
    cinds=[0,2,3,4,7,8,9,10]

    ms = []
    two_means = []
    ward =[]
    spectral = []
    dbscan = []
    average_linkage = []
    affinity_propagation = []
    birch = []
    gmm = []
    bgmm = []
    hdb = []

    cinds_all = (
        ('KMeans', two_means),
        ('AffinityPropagation', affinity_propagation),
        ('MeanShift', ms),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        ('Birch', birch),
        ('HDBSCAN', hdb),
        ('GaussianMixture', gmm),
        ('BGaussianMixture', bgmm)
    )

    clustering_algorithms = [cinds_all[j] for j in cinds]

    bars_de2 = np.mean(np.asarray(bars_de),axis=0)
    bars_se2 = np.mean(np.asarray(bars_se),axis=0)
    bars_pc2 = np.mean(np.asarray(bars_pc),axis=0)

    indeces = np.argsort(bars_pc2)

    bars_se2 = bars_se2[indeces]
    bars_de2 = bars_de2[indeces]

    bars_pc2.sort()

    # vs = np.asarray(vs)
    bars = []
    colors = []
    n_components_range = range(len(bars_pc2))
    # cv_types = [item[1] for item in datasets2]


    # color_iter = cycle(['navy', 'turquoise', 'cornflowerblue',
    #                               'darkorange','k'])
    # # Plot the BIC scores
    # plt.figure(figsize=(8, 6))
    # plt.rcParams['axes.facecolor'] = (1,1,1)            
    # spl = plt.subplot(1, 1, 1)

    # for i, (cv_type, pcolor) in enumerate(zip(cv_types, color_iter)):
    #     xpos = np.array(n_components_range) + 0.166 * (i - 2)
    #     bars.append( vs[i * len(n_components_range):
    #                                   (i + 1) * len(n_components_range)])

    # # import ipdb; ipdb.set_trace()
    [colors.append(colourblind(col)) for col in n_components_range]

    # bars2 = np.mean(np.asarray(bars),axis=0)

    # indeces = np.argsort(bars2)
    # #         import ipdb; ipdb.set_trace()
    # bars2.sort()
    # bars2 = np.concatenate(bars_se2,bars_de2,bars_pc2)

    clustering_algorithms = [clustering_algorithms[i] for i in indeces]

    plt.bar(range(0,len(bars_pc2)*3,3),bars_se2,color=colourblind(1))
    plt.bar(range(1,len(bars_pc2)*3,3),bars_de2,color=colourblind(2))
    plt.bar(range(2,len(bars_pc2)*3,3),bars_pc2,color=colourblind(3))
    plt.xticks(range(1,len(bars_pc2)*3,3),[item[0] for item in clustering_algorithms],rotation = 45, ha="right")
    plt.xticks()
    plt.ylim([0, 1])
    plt.title('V Averaged over ' +title+ ' Materials')
    # xpos = np.mod(vs.argmax(), len(n_components_range)) + .65 +\
    #     .16 * np.floor(vs.argmax() / len(n_components_range))
    # plt.text(xpos, vs.min() * 0.97 + .03 * vs.max(), '*', fontsize=14)
    plt.xlabel('Algorithm')
    plt.ylabel('V-score')
    plt.legend(leg)  
    plt.tight_layout()
    
    plt.savefig('{}.png'.format(title))