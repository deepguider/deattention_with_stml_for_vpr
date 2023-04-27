import argparse
import os
import json
from ipdb import set_trace as bp

def get_parser():
    ## Init.
    parser = argparse.ArgumentParser(description='pytorch-myImageRetrieval')

    ## User defined (begin)
    parser.add_argument('--verbose', action='store_true', help='Print internal processing message.')
    parser.add_argument('--reuse_cache_for_debug', action='store_true', help='For quick debugging, resuse prebuilt cache file. Do not use this option for normal task.')
    parser.add_argument('--deatt_padding_mode', type=str, default='reflect', help="The reflect mode is better than zeros padding mode because it does not create boundary artifact.", choices=['zeros', 'reflect'])
    parser.add_argument('--deattention_version', type=int, default=1, help='Deattention version 1 : feature = feature*attention(feature), version 2: feature = feature + feature*attention(feature), version 3: 32, 32, 32 at CRN ')
    parser.add_argument('--deattention', action='store_true', help='Deattention by segmentation (Masking car, human features).')
    parser.add_argument('--deattention_auto', action='store_true', help='Deattention by segmentation (Masking car, human features). Choose weight of object automatically.')
    parser.add_argument('--deatt_weighted_mask', action='store_true', help='Deattention by segmentation (Masking car, human features with weighting from database class histogram).')
    parser.add_argument('--deatt_category_list', nargs='+', default=["vehicle", "human"], help='List of deattention category among ["vehicle", "human", "nature", "sky"]', required=False)
    parser.add_argument('--add_segmask_to_input_ch4', action='store_true', help='Deattention at the input image by putting segmentation (Masking car, human features) into 4-th channel of input image.')
    parser.add_argument('--write_heatmap', action='store_true', help='Write heat-map to image files.')
    parser.add_argument('--deatt_loss_is_bce', action='store_true', help='Loss function to train deattention layer. Default(false) is MSE. BCE when this option in enabled(True).')
    parser.add_argument('--internal_result_path', type=str, default='.', help='Result directory where some internal results are stored. It will be updated in writer_init() automatically.')
    parser.add_argument('--heatmap_result_dir', type=str, default='heatmap', help='Result directory where some internal results are stored. It will be updated in writer_init() automatically.')
    parser.add_argument('--w_deatt_loss', type=float, default=0.001, help='Balancing weight of deatt loss for deattention.')
    parser.add_argument('--seg_ckpt', type=str, default='./networks/MobileNet/pretrained/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')    
    parser.add_argument('--add_clutter_train', action='store_true', help='Add clutters into test image, in which cropped segmented objects are overwrited into image in __getitem__() of dataset')
    parser.add_argument('--add_clutter_test_q', action='store_true', help='Add clutters into query of test image, in which cropped segmented objects are overwrited into image in __getitem__() of dataset. When q is enabled, its baseline Re@1 is 0.77')
    parser.add_argument('--add_clutter_test_db', action='store_true', help='Add clutters into db of test image, in which cropped segmented objects are overwrited into image in __getitem__() of dataset. When q and db are enabled, its baseline Re@1 is 0.70')
    parser.add_argument('--add_clutter_iteration', type=int, default=3, help='In add_clutter, a number indicating how may iterations to repeat for a single image.')
    parser.add_argument('--write_add_clutter_image', action='store_true', help='It write cluttered image into file.')
    parser.add_argument('--do_not_process_only_for_writing_image', action='store_true', help='Debug mode. Only to get cluttered image not to do test or train process.')

    parser.add_argument('--remove_clutter_input', action='store_true', help='Forcely set clutter region of input image to some values, zero, one, random value, and mosaic (gaussian blurring).')
    parser.add_argument('--remove_clutter_mode', type=str, default='zero', help='Forcely set clutter region of input image to this mode.', choices=['zero', 'one', 'random', 'mosaic'])

    parser.add_argument('--which_cuda', type=str, default='cuda', choices=["cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"], help='Specify cuda to run. cuda is for all cuda. cuda:N means using a N-th single cuda.')
    parser.add_argument('--recall_radius', type=float, default=0.0, help='Default is 25.0. Tolerance range of position error in meters when calculating recall.')
    parser.add_argument('--test_write_image', action='store_true', help='Write result image file during test')
    parser.add_argument('--write_data_distribution_in_train', action='store_true', help='Write distance distribution of q-p and q-n.')

    parser.add_argument('--save_ckpt_without_train', action='store_true', help='Write pre-trained weight by Imagenet without any train.')

    ## For loss
    parser.add_argument('--tml2_version', type=int, default=1, help='Custom TMLoss function, ver2 : loss = m - d(p,n), ver1:  loss = d(q,p) + m + pn_margin - d(q,n) - d(p,n)', choices=[1, 2])
    parser.add_argument('--tml2', action='store_true', help='Custom TMLoss function : loss = d(q,p) + m - d(q,n).')
    parser.add_argument('--tml2_pn_margin', type=float, default=1.4, help='Custom TMLoss function witht added positive and negative distances.')
    parser.add_argument('--loss_statistics', action='store_true', help='Calculate mean and std of d_qp, d_qn, d_pn for a Paper.')
    parser.add_argument('--dataloader_margin', type=float, default=0.1, help='Margin for dataloader. Default=0.1')

    ## For other attention
    parser.add_argument('--crn_attention', action='store_true', help='CRN attention by Kim et.al.')
    parser.add_argument('--bam_attention', action='store_true', help='BAM attention')
    parser.add_argument('--cbam_attention', action='store_true', help='CBAM attention')
    parser.add_argument('--senet_attention', action='store_true', help='SENET(Squeeze and Excitation) attention')
    parser.add_argument('--ch_attention', action='store_true', help='Simple channel attention after Deattention. Recall was degraded from 0.87 to 0.85 with ch_attention')
    parser.add_argument('--ch_eca_attention', action='store_true', help='ECA channel attention after Deattention. Recall was degraded from 0.87 to 0.85 with ch_attention')
    parser.add_argument('--ch_eca_attention_k_size', type=int, default=3, help='kernel size of ECA channel attention.')

    ## For post-processing
    parser.add_argument('--rerank', action='store_true', help='Do rerank after netvlad.')
    parser.add_argument('--rerank_ratio', type=float, default=0.3, help='cost ratio = (best - second best) / best. Default=0.3')
    parser.add_argument('--save_feature', action='store_true', help='Save features to Feat.pickle.')
    parser.add_argument('--load_feature', action='store_true', help='Load features from Feat.pickle.')
    parser.add_argument('--feature_fname', type=str, default='Feat.pickle')    
    parser.add_argument('--save_test_result_avi_with_map', action='store_true', help='Save matched image on a map')
    parser.add_argument('--save_test_result_avi', action='store_true', help='Save matched image to [dataset]_[split]_Success/Fail.avi')
    parser.add_argument('--save_test_result_avi_disable_text', action='store_true', help='Disable to write text information such as image name on avi files. In default mode, text is written on avi files.')
    parser.add_argument('--pca_whitening', action='store_true', help='Use PCA whitening.')
    parser.add_argument('--pca_whitening_mode', type=int, default=11, help='PCA whitening mode. 011 get the best. choices are [000, 001, 010, 011, 100, 101, 110, 111], where each value means (whiten=0/1, inScale=0/1, outNorm=0/1)', choices=[0, 1, 10, 11, 100, 101, 110, 111])
    parser.add_argument('--pca_dim', type=int, default=0, help='PCA dimension. Default 0 means as is.', choices=[256, 512, 1024, 4096])

    ## Write result
    parser.add_argument('--matched_result_dir', type=str, default='matched_result_dir')    

    ## For dg project report
    parser.add_argument('--conf_thre', type=float, default=0.70, help='Accuracy threshold of confidence.')
    parser.add_argument('--report_verbose', action='store_true', help='Print only recall@1 and accuracy@3 for deepguider report.')

    ## User defined (end)
    
    ## Defaults
    parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster', 'class_statics'])
    parser.add_argument('--batchSize', type=int, default=4, 
            help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
    parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
    parser.add_argument('--cacheRefreshRate', type=int, default=1000, 
            help='How often to refresh cache, in number of queries. 0 for off')
    parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
            help='manual epoch number (useful on restarts)')
    parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
    parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate.')
    parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
    parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
    parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
    parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
    parser.add_argument('--dataPath', type=str, default='checkpoints/data/', help='Path for centroid data.')
    parser.add_argument('--runsPath', type=str, default='checkpoints/runs/', help='Path to save runs to.')
    parser.add_argument('--testPath', type=str, default='checkpoints/test/', help='Path to save test to.')
    parser.add_argument('--endPath', type=str, default='checkpoints', help='Last end(sub) path to save checkpoints. Do not edit this.')
    parser.add_argument('--savePath', type=str, default='checkpoints', help='Full path to save checkpoints.')  # It may be overwritten by join()
    try:
        parser.add_argument('--cachePath', type=str, default=os.environ['TMPDIR'], help='Path to save cache to.')
    except:
        parser.add_argument('--cachePath', type=str, default='/mnt/ramdisk', help='Path to save cache to.')
    parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
    parser.add_argument('--ckpt', type=str, default='latest', 
            help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
    parser.add_argument('--evalEvery', type=int, default=1, 
            help='Do a validation set run, and save, every N epochs.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
    parser.add_argument('--dataset', type=str, default='pittsburgh', 
            help='Dataset to use', choices=['pittsburgh', 
                'pittsburgh_3k',
                'pittsburgh_6k',
                'pittsburgh_9k',
                'pittsburgh_12k',
                'pittsburgh_15k',
                'pittsburgh_18k',
                'pittsburgh_21k',
                'pittsburgh_24k',
                'pittsburgh_27k',
                'tokyo247', 'tokyotm', 'tokyoTM', 'roxford5k', 'rparis6k', 'dg_daejeon', 'dg_bucheon', 'dg_seoul'])
    parser.add_argument('--dataset_map_idx', type=int, default=0, help = '')            
    parser.add_argument('--arch', type=str, default='vgg16', 
            help='basenetwork to use', choices=['vgg16', 'alexnet'])
    parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
    parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
            choices=['netvlad', 'max', 'avg', 'gem'])
    parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
    parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
    parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', 
            choices=['test', 'test250k', 'train', 'train250k', 'val'])
    parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')

    ## Parsing args to opt
    opt = parser.parse_args()
    opt = resume_parameters(opt, parser)  # It is called in utils/misc/resume_ckpts()
    return opt

def resume_parameters(opt, parser):
    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum',
            'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
            'margin', 'dataloader_margin', 'tml2_pn_margin', 'tml2_version', 'tml2', 'seed', 'patience']
    verbose = opt.verbose
    if opt.resume:
        flag_file = os.path.join(opt.resume, 'checkpoints', 'flags.json')
        if os.path.exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {'--'+k : str(v) for k,v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del: del stored_flags[flag]
                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                if verbose:
                    print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)
    if verbose:
        print(opt)
    return opt
