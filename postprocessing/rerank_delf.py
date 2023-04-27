from postprocessing import delf_image_matcher
import os
import copy
import numpy as np
from ipdb import set_trace as bp

class rerank():
    def __init__(self, eval_set=None, dataset=None, device='/device:GPU:1'):  # If you met lack of memory. change this to GPU:1
        ## Init. DELF
        self.imatch = delf_image_matcher.delf_image_matcher(device)
        if eval_set is not None:
            self.init_dataset(eval_set.dbStruct, None, None)
        if dataset is not None:
            self.init_dataset(None, dataset.queries_dir, dataset.db_dir)
        self.skip_cnt = 0
        self.reranked_cnt = 0
        #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

    def init_dataset(self, dbStruct=None, queries_dir=None, db_dir=None):
        if dbStruct is not None:
            self.dbStruct = dbStruct
        if queries_dir is not None:
            self.queries_dir = queries_dir
        if db_dir is not None:
            self.db_dir = db_dir

    def get_dbq_flist(self, qIx, pred):
        q_dir = self.queries_dir
        db_dir = self.db_dir
        qimg = os.path.join(q_dir, self.dbStruct.qImage[qIx])
        dblist = []
        for i, idx in enumerate(pred,1):
            predname = os.path.join(db_dir, self.dbStruct.dbImage[idx])
            dblist.append(predname)
            if i >= 100:  # Let's set max to 100.
                break
        return qimg, dblist

    def confidence_valid_check(self, confidence, ratio=0.3):  # ratio > 0.0 ( ratio between best and second best )
        conf = copy.copy(confidence)
        conf.sort()
        best = conf[-1]
        second_best = conf[-2]
        if best > 0 :
            cost_ratio = ( best - second_best ) / best
        else:
            return False

        if cost_ratio >= ratio:
            return True
        else:
            return False

    def get_reranked_cnt(self):
        return self.reranked_cnt

    def run(self, qIx, pred, search_range=0, ratio=0.3, disp_en=False):
        reranked = False
        if search_range == 0:
            search_range = len(pred)  # 20
        qimg, dblist = self.get_dbq_flist(qIx, pred[:search_range])
        pred_rerank=None
        try:
            inliers = []  # Similarity
            q_img = delf_image_matcher.download_and_resize(qimg)
            self.imatch.set_image2(q_img)
            for db_path in dblist:
                db_img = delf_image_matcher.download_and_resize(db_path)
                inlier = self.imatch.match_images(db_img, None, knn_mode='knn', ransac_en=True, disp_en=True)
                inliers.append(inlier)

            #if self.confidence_valid_check(inliers, ratio):  # default
            if True:  # for test
                ## Display result
                if disp_en:
                    db_img = delf_image_matcher.download_and_resize(dblist[pred_rerank])
                    inlier = self.imatch.match_images(db_img, None, knn_mode='knn', ransac_en=True, disp_en=disp_en)
                    bp()
                    q_img.show();db_img.show()

                # Swap best 1
                pred_rerank = np.argmax(inliers)
                conf_rerank = inliers[pred_rerank]

                tmp = pred[0]
                pred[0] = pred[pred_rerank]
                pred[pred_rerank] = tmp
                self.reranked_cnt += 1
                reranked = True
        #if reranked:
        #    print("Reranked")
        except:
            self.skip_cnt = self.skip_cnt + 1
            print(" # [{}] Skip Reranking due to lack of feature".format(self.skip_cnt))
        return pred, reranked
        #return pred, reranked, pred_rerank

