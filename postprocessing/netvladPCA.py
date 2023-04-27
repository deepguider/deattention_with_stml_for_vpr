import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib # dump, load
import faiss
import numpy as np
from ipdb import set_trace as bp

class netvladPCA():
    ## To do : I didn't implement completely whiten function of netvlad paper.
    def __init__(self, dim=128, whiten=False, inScale=True, outNorm=True, random_seed=123):
        self.whiten = whiten  # It's different from sklearn's whiten
        self.dim = dim 
        self.pca = PCA(n_components=dim, whiten=False, random_state=random_seed)
        self.feat_out = []
        self.inScale = inScale
        self.outNorm = outNorm
        if self.whiten:  # Eq5 in https://hal.inria.fr/hal-00722622v2/document
            print("Warning : I have not implemented **Whiten** function yet.")

    def fit(self, feat):  # feat is reference feature, ex) db_feat
        feat = feat.astype('float32')  # size * depth : 10000 * 32768 
        print('====> Training PCA : {} to {}'.format(feat.shape[-1], self.dim))
        if self.inScale:
            scaled = StandardScaler().fit_transform(feat)
        else:
            scaled = feat
        self.pca.fit(scaled)
        if self.whiten:  # Eq5 in https://hal.inria.fr/hal-00722622v2/document
            self.pca.components_ = np.dot(np.diag(1/np.sqrt(self.pca.singular_values_ + 1e-9)), self.pca.components_)

    def transform(self, feat): # feat is current feature, ex) q_feat
        feat = feat.astype('float32')  # size * depth : 10000 * 32768 
        print('====> Transform features using PCA : {} to {}'.format(feat.shape[-1], self.dim))
        if self.inScale:
            scaled = StandardScaler().fit_transform(feat)
        else:
            scaled = feat
        feat_pca = self.pca.transform(scaled)

        if self.whiten:  # Eq5 in https://hal.inria.fr/hal-00722622v2/document
            feat_pca = feat_pca/np.linalg.norm(feat_pca)

        if self.outNorm:   # recall increase about +5 %p when using outNorm=True
            feat_pca = cv2.normalize(feat_pca, None, 1.0, 0.0, cv2.NORM_L2)
        return feat_pca

    def save_pca_model(self):
        joblib.dump(self.pca, 'pca_dim{}.joblib'.format(self.dim))

    def load_pca_model(self):
        self.pca = joblib.load('pca_dim{}.joblib'.format(self.dim))

    def get_pca_model(self):
        return self.pca

# from netvladPCA import netvladPCA
if __name__ == '__main__':
    ## Make random input data
    np.random.seed(100)
    in_dim = 32768
    pca_dim = 256 
    dbFeat = np.random.rand(1000, in_dim)
    qFeat = np.random.rand(100, in_dim)

    ## PCA

    #mPCA = netvladPCA(dim=pca_dim, whiten=False, inScale=False, outNorm=False)  #000
    #mPCA = netvladPCA(dim=pca_dim, whiten=False, inScale=False, outNorm=True)   #001
    #mPCA = netvladPCA(dim=pca_dim, whiten=False, inScale=True, outNorm=False)   #010
    mPCA = netvladPCA(dim=pca_dim, whiten=False, inScale=True, outNorm=True)    #011  # best recall
    #mPCA = netvladPCA(dim=pca_dim, whiten=True, inScale=False, outNorm=False)   #100
    #mPCA = netvladPCA(dim=pca_dim, whiten=True, inScale=False, outNorm=True)    #101
    #mPCA = netvladPCA(dim=pca_dim, whiten=True, inScale=True, outNorm=False)    #110
    #mPCA = netvladPCA(dim=pca_dim, whiten=True, inScale=True, outNorm=True)     #111
    mPCA.fit(dbFeat)
    new_dbFeat = mPCA.transform(dbFeat)
    new_qFeat = mPCA.transform(qFeat)

    print(" {} ==> {} ".format(dbFeat.shape[-1], new_dbFeat.shape[-1]))

    ## kNN
    #k = 20  # top 20
    #pool_size = dbFeat.shape[-1]
    #faiss_index = faiss.IndexFlatL2(pool_size)
    #faiss_index.add(dbFeat)
    #distances, predictions = faiss_index.search(qFeat, k )
