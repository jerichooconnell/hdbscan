

import os, shutil
import itertools
import imageio

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from scipy import linalg

from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import NMF, FastICA, PCA
from sklearn.metrics import homogeneity_score,homogeneity_completeness_v_measure
from sklearn import mixture
from skimage.restoration import inpaint
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


import scipy.ndimage as ndimage
import scipy.spatial as spatial
import scipy.misc as misc
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Ellipse

class I_gmm:
    def __init__(self):
        self.XY = None
        
    def plot_cov(self,means, covariances,ct):
        if ct == 'spherical':
            return
        color_iter = itertools.cycle(['navy', 'navy', 'cornflowerblue', 'gold',
                                  'darkorange'])
        ax =plt.gca()
        for i, (mean, covar, color) in enumerate(zip(
                means, covariances, color_iter)):
                
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            alpha = 0.2
            ell = Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(alpha)
            ax.add_artist(ell)
            ell = Ellipse(mean, v[0]*4, v[1]*2, 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(alpha)
            ell = Ellipse(mean, v[0]*2, v[1]*2, 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(alpha)
            ax.add_artist(ell)            
#             multi_normal = multivariate_normal(mean[0:2],u[0:2])
#             ax.contour(np.sort(X[:,0]),np.sort(X[:,1]),
#                    multi_normal.pdf(self.XY).reshape(X.shape[0],X.shape[0]),
#                    colors='black',alpha=0.3)
            ax.scatter(mean[0],mean[1],c='grey',zorder=10,s=100)        
        
    def iterative_gmm(self,dataset = 'bb',fake = True,mode2 = 'gmm',mode=[],binary = False,im_dir = './images/',savegif = False,title ='temp',bic_thresh = 0,maxiter = 40,nc =5,v_and_1 = False,thresh = 0.9,cov=[],n_components=2,covt='spherical',ra=False,red = 'pca'):

        '''
        dataset: string
        The filename of the material something like 'bb','pp'
        fake: bool
        Whether or not the data is fake, if it is not it will be cropped
        mode: str
        'fraction' will reduce the input to a combination of the relative signals
        e.g. bin1 - bin0/sum
        binary: bool
        Whether or not to show the output as binary or not
        nc: int
        pca components
        '''
        # Clear the imagedir
        if savegif:
            folder = im_dir
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        if ra:
            arrowsU = []
            arrowsV = []        
        bic0 = np.infty
        itern = 0
        inds = [2,3,4]
        label_true = loadmat('2'+dataset+'_mask')['BW'] 
        
        if mode == 'de':
            X1 = loadmat('all'+dataset)['Z']
            for jj in range(0,6):
                r2 = np.reshape(X1[:,jj], (20,68), order="F")
                X1[:,jj] = np.reshape(gaussian_filter1d(r2,sigma=1),20*68,order="F")            
            X1 = np.column_stack((np.sum(X1[:,1:3],1),np.sum(X1[:,4:],1)))
        elif mode == 'se':
            X1 = loadmat('all'+dataset)['Z'][:,0]
            r2 = np.reshape(X1, (20,68), order="F")
            X1 = np.reshape(gaussian_filter1d(r2,sigma=1),20*68,order="F")            
            X1 = np.reshape(X1, (20*68,1), order="F")
        elif mode == 'all':
            X1 = loadmat('all'+dataset)['Z']
            for jj in range(0,6):
                r2 = np.reshape(X1[:,jj], (20,68), order="F")
                X1[:,jj] = np.reshape(gaussian_filter1d(r2,sigma=1),20*68,order="F")
            if red == 'pca':
                X1 = PCA(n_components=nc).fit_transform(X1)            
        elif mode == 'small':
            X1 = loadmat('small_'+dataset)['Z'] 
        elif mode == 'gauss':
            X1 = loadmat('all'+dataset)['Z']
            for jj in range(0,6):
                r2 = np.reshape(X1[:,jj], (20,68), order="F")
                X1[:,jj] = np.reshape(gaussian_filter1d(r2,sigma=1),20*68,order="F")
        else:
            X1 = loadmat('all'+dataset)['Z'][:,inds]
            if red == 'pca':
                X1 = PCA(n_components=nc).fit_transform(X1)
                
        if red == 'ica':
            X1 = FastICA(n_components=nc,whiten=True).fit_transform(X1)
            covt = 'full'
        if red == 'icapca':
            X1 = FastICA(n_components=5).fit_transform(X1)
            X1 = PCA(n_components=nc).fit_transform(X1)                
        if red == 'tsne':
            X1 = TSNE(n_components=nc).fit_transform(X1)
        if red == 'nmf':
            X1 = NMF(n_components=nc).fit_transform(X1)  
            covt = 'full'
        if red == 'spec':
            X1 = SpectralEmbedding(n_components=3).fit_transform(X1)
            
        if not fake:
            X1 = X1[400:,:].copy()
            label_true = label_true[400:].copy()
            length = 48
        else:
            length = 68

        # This is code for just looking at the ratio of the bins
        if mode == 'fraction':
            # initialize vector for fraction
            X2 = np.zeros([X1.shape[0],int(scipy.special.comb(5,2))])
            result = [x for x in itertools.combinations(np.arange(5),2)]
            for jj in range(0,X2.shape[1]):
                r2 = np.reshape(abs(X1[:,result[jj][0]] - X1[:,result[jj][1]])/abs(X1[:,result[jj][0]] + X1[:,result[jj][1]]), (20,length), order="F")
                X2[:,jj] = np.reshape(gaussian_filter1d(r2,sigma=1),20*length,order="F")
            ind = np.argsort(np.mean(X1,0))
        #     X1 = X2[:,ind[:4]]
        else:
            for jj in range(0,X1.shape[1]):
                r2 = np.reshape(X1[:,jj], (20,length), order="F")
                X1[:,jj] = np.reshape(gaussian_filter1d(r2,sigma=1),20*length,order="F")
        ct = 'full'
        
        if mode2 == 'bgmm':
            bgmm = mixture.BayesianGaussianMixture(
                    n_components=n_components, covariance_type=covt)
        elif mode2 == 'kmeans':
            km = KMeans(n_clusters=n_components)
        
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=covt)
        gmm1 = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='full')

        ims = []
        
        
        # Do the PCA decomposition
#         if red == 'pca':
#             X1 = PCA(n_components=nc).fit_transform(X1)
        
        X3 = X1[:,0:2].copy()
        
        fig = plt.figure(figsize=(10,10))
        bics = []        
        for ii in range(0,maxiter):

            X = X1.copy()

            if mode2 == 'gmm':
                y_pred = gmm.fit_predict(X)
            elif mode2 == 'bgmm':
                y_pred = bgmm.fit_predict(X)
                y_ff = gmm.fit(X)
            elif mode2 == 'kmeans':
                y_pred = km.fit_predict(X)
                y_ff = gmm.fit(X)
            
            y_ff1 = gmm1.fit(X)
            
            # if I should show the vmeasure
            if v_and_1:

                homo1,comp1,vs1 = homogeneity_completeness_v_measure(
                    label_true.squeeze(), y_pred)

                bic = gmm.aic(X)
                bic1 = gmm1.aic(X)
                if savegif:
                    print(vs1,itern,bic,bic1)
            else:
                bic = gmm.aic(X)
                if savegif:
                    print(bic)

            # Stop if bic is lower
            if bic - bic0 < bic_thresh:
                bic0 = bic
            else:
                print('BIC got higher')
                break
            if savegif:
                print(bic)
            
            # map the bad values to zero
            for kk in range(n_components):
                temp = X[y_pred == kk,:]

                if cov == 'robust':
                    robust_cov = MinCovDet().fit(temp)
                else:
                    robust_cov = EmpiricalCovariance().fit(temp)
                
                # Calculating the mahal distances
                robust_mahal = robust_cov.mahalanobis(
                    temp - robust_cov.location_) ** (0.33)
                
                if thresh < 1:
                    temp[robust_mahal > robust_mahal.min() + (robust_mahal.max()-robust_mahal.min()) *thresh] = 0
                else:
#                     import pdb; pdb.set_trace()
                    temp[robust_mahal > np.sort(robust_mahal)[-thresh]] = 0
                    
                X[y_pred == kk,:] = temp

            mask_one = X[:,0] == 0

            if y_pred[3] == 0:
                # Map top to zero if it is the wrong combo
                y_pred = y_pred + 1
                y_pred[y_pred == n_components] = 0

            m_reshape = np.reshape(mask_one, (20,length), order="F")

            if itern == 0:
                y_0 = y_pred
            
            # Plotting functions
            if savegif:
                ax0 = fig.add_subplot(111)

                a = -(y_pred - label_true.squeeze())
                y_reshape = np.reshape(a, (20,length), order="F")

                colorz = ['b','r','g','m']
                for jj,color in zip(range(a.min(),a.max()+1),colorz):
                    print(jj)
                    b = a == jj
                    b = [i for i, x in enumerate(b) if x]
                    if jj == 0:
                        c = b
                    ax0.scatter(X1[b,0],X1[b,1],c=colorz[(jj-a.min())])

                ax0.set_title('New Method')
                self.plot_cov(gmm1.means_, gmm1.covariances_,ct='full')

                if itern == 0:
                    axes = plt.gca()
                    ylim = axes.get_ylim()
                    xlim = axes.get_xlim() 

                ax0.set_xlim(xlim)
                ax0.set_ylim(ylim)

                ax0.scatter(X3[:,0],X3[:,1],c='k',alpha = 0.1)
                plt.text(.5*xlim[-1], ylim[0] + .005,'bad pts = {}'.format(format(len(c),"03")))     
                ax3 = plt.axes([.22, .15, .15, .1])
                bics.append(bic)
                plt.plot(bics)
                plt.yticks([])
                plt.xlabel('iteration')
                plt.ylabel('BIC')
                ax2 = plt.axes([.25, .55, .6, .4], facecolor='y')

                if binary:
                    plt.imshow(y_reshape,cmap='brg')
                else:
                    plt.imshow(np.reshape(X1[:,0], (20,length), order="F"))

                plt.title('Image Space')
                plt.xticks([])
                plt.yticks([])

                if savegif:
                    plt.savefig(im_dir + '{}.png'.format(format(itern, "02")))

                itern += 1

                i, j = np.where(m_reshape == True)

    #             if binary:
    #                 plt.imshow(y_reshape,cmap='brg')
    #             else:
                plt.scatter(j,i,marker='x',c='k')

    #             import pdb; pdb.set_trace()
                d = [i for i, x in enumerate(mask_one) if x]
                ax0.scatter(X1[d,0],X1[d,1],marker='x',c='k')


                if savegif:
                    plt.savefig(im_dir + '{}.png'.format(format(itern, "02")))

                itern += 1
            
            X2 = X1.copy()
            # Inpainting the zeros
            r2 = np.reshape(X1, (20,length,X1.shape[1]), order="F")
            X1 = np.reshape(inpaint.inpaint_biharmonic(
                r2,m_reshape,multichannel=True),
                            (20*length,X1.shape[1]),order="F")
            
            if savegif:
                ax0.plot([X2[d,0],X1[d,0]],[X2[d,1],X1[d,1]],'r')

                if ra:
                    arrowsU.append([X2[d,0],X1[d,0]])
                    arrowsV.append([X2[d,1],X1[d,1]])


                if binary:
                    plt.imshow(y_reshape,cmap='brg')
                else:
                    plt.imshow(np.reshape(X1[:,0], (20,length), order="F"))
                plt.savefig(im_dir + '{}.png'.format(format(itern, "02")))
                plt.clf()
            
            

    #         X_old = X.copy()
    #             np.save('bb',y_reshape)

        #     plt.figure()
        #     robust_mahal1.sort()
        #     plt.plot(robust_mahal1)
        #     plt.plot(250,robust_mahal1.max()*.87,'r*')
        #     plt.savefig('./images2/{}.png'.format(format(itern, "02")))  

        #     plt.figure()
        #     robust_mahal2.sort()
        #     plt.plot(robust_mahal2)
        #     plt.plot(250,robust_mahal2.max()*.87,'r*')
        #     plt.savefig('./images3/{}.png'.format(format(itern, "02")))  
            itern += 1

        if savegif:
            fig = plt.figure(figsize=(10,10))
            ax0 = fig.add_subplot(111)  

            colorz = ['b','r','g','m']
            for jj,color in zip(range(a.min(),a.max()+1),colorz):
                print(jj)
                b = a == jj
                b = [i for i, x in enumerate(b) if x]
                if jj == 0:
                    c = b
                ax0.scatter(X1[b,0],X1[b,1],c=colorz[(jj-a.min())])

            self.plot_cov(gmm1.means_, gmm1.covariances_,ct)
            ax0.scatter(X3[:,0],X3[:,1],c='k',alpha = 0.1)

            ax0.set_title('New Method')
            ax0.set_xlim(xlim)
            ax0.set_ylim(ylim)
            plt.text(.5*xlim[-1], ylim[0] + .005,'bad pts = {}'.format(format(len(c),"03")))      

            r = np.reshape(y_pred, (20,length), order="F")
            if binary:
                self.animate(y_reshape)
            else:
                self.animate(y_reshape, im=np.reshape(X1[:,0], (20,length), order="F"))
            ax3 = plt.axes([.22, .15, .15, .1])
            bics.append(bic)
            plt.plot(bics)
            plt.yticks([])
            plt.xlabel('iteration')
            plt.ylabel('BIC')
            plt.savefig(im_dir + '{}.png'.format(format(itern, "02")))


            plt.figure()
            plt.imshow(r)
            plt.xticks([])
            plt.yticks([])  
            plt.figure()

            r0 = np.reshape(y_0, (20,length), order="F")
            plt.imshow(r - r0)
            plt.xticks([])
            plt.yticks([]) 
        else:
            return vs1

        if savegif:
            # save gif
            files = os.listdir('./images')
            images = []
            for filename in files:
                images.append(imageio.imread('./images/'+filename))
            imageio.mimsave(title + '.mp4', images,fps=1)
            imageio.mimsave(title + '.gif', images)


        if ra:
            return arrowsU,arrowsV
    def find_paws(self,data, smooth_radius = 1, threshold = 0.0001):
        # https://stackoverflow.com/questions/4087919/how-can-i-improve-my-paw-detection
        """Detects and isolates contiguous regions in the input array"""
        # Blur the input data a bit so the paws have a continous footprint 
        data = ndimage.uniform_filter(data, smooth_radius)
        # Threshold the blurred data (this needs to be a bit > 0 due to the blur)
        thresh = data > threshold
        # Fill any interior holes in the paws to get cleaner regions...
        filled = ndimage.morphology.binary_fill_holes(thresh)
        # Label each contiguous paw
        coded_paws, num_paws = ndimage.label(filled)
        # Isolate the extent of each paw
        # find_objects returns a list of 2-tuples: (slice(...), slice(...))
        # which represents a rectangular box around the object
        data_slices = ndimage.find_objects(coded_paws)
        return data_slices

    def animate(self,frame,im = None):
        """Detects paws and animates the position and raw data of each frame
        in the input file"""
        # With matplotlib, it's much, much faster to just update the properties
        # of a display object than it is to create a new one, so we'll just update
        # the data and position of the same objects throughout this animation...

        # Since we're making an animation with matplotlib, we need 
        # ion() instead of show()...
        fig = plt.gcf()
        ax = plt.axes([.25, .55, .6, .4], facecolor='y')
        plt.axis('off')

        # Make an image based on the first frame that we'll update later
        # (The first frame is never actually displayed)
        if im is None:
            plt.imshow(frame,cmap='brg')
        else:
            plt.imshow(im)
        plt.title('Image Space')

        # Make 4 rectangles that we can later move to the position of each paw
        rects = [Rectangle((0,0), 1,1, fc='none', ec='red') for i in range(4)]
        [ax.add_patch(rect) for rect in rects]


        # Process and display each frame

        paw_slices = self.find_paws(frame)

        # Hide any rectangles that might be visible
        [rect.set_visible(False) for rect in rects]

        # Set the position and size of a rectangle for each paw and display it
        for slice, rect in zip(paw_slices, rects):
            dy, dx = slice
            rect.set_xy((dx.start, dy.start))
            rect.set_width(dx.stop - dx.start + 1)
            rect.set_height(dy.stop - dy.start + 1)
            rect.set_visible(True)
