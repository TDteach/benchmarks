from enum import Enum
class Net_Mode(Enum):
    NORMAL = 0
    TRIPLE_LOSS = 1
    BACKDOOR_DEF = 2

class Data_Mode(Enum):
    NORMAL = 0
    POISON = 1
    SINGLE_CLASS = 2

class Load_Mode(Enum):
    NORMAL = 0
    ALL = 1
    BOTTOM_AFFINE = 2

class Fix_Level(Enum):
    NONE = 0
    AFFINE = 1
    BOTTOM = 2
    BOTTOM_AFFINE = 3

class Options:
    max_steps = None
    #batch_size= 24 # 24 for resnet101
    #batch_size= 96 # 96 for Net_Mode.BACKDOOR_DEF
    batch_size= 128 # 128 for gtsrb
    num_epochs = 30
    num_gpus = 1
    num_loading_threads=8

    shuffle = True

    scale_size = 300
    # crop_size = 128
    crop_size = 32
    mean = 127.5

    home_dir = '/home/tdteach/'
    log_dir = home_dir+'logs/'
    data_dir = home_dir+'data/GTSRB/train/Images/'
    # data_dir = home_dir+'data/MF/train/'
    # data_dir = home_dir+'data/imagenet/'

    checkpoint_folder = home_dir+'data/checkpoint/'

    #for MegaFace
    caffe_model_path = home_dir+'data/caffe_models/ResNet_101_300K.npy'
    n_landmark=68
    meanpose_filepath=data_dir+'lists/meanpose68_300x300.txt'
    image_folders=[data_dir+'tightly_cropped/']
    #list_filepaths=[data_dir+'lists/lists_wedge/list_10_wedge.txt']
    #landmark_filepaths=[data_dir+'lists/lists_wedge/landmarks_10_wedge.txt']
    list_filepaths=[data_dir+'lists/list_all.txt']
    landmark_filepaths=[data_dir+'lists/landmarks_all.txt']

    solid_pattern = home_dir+'workspace/backdoor/solid_rd.png'



    net_mode = 'normal' #normal backdoor_def
    # for Net_Mode.BACKDOOR_DEF
    loss_lambda = 0.1 # for gtsrb
    #loss_lambda = 0.01 # for megaface
    # for Net_Mode.TRIPLE_LOSS
    use_triplet_loss = False
    size_triplet_slice = 4
    num_slices_one_batch = batch_size // size_triplet_slice

    build_level = 'logits'

    selected_training_labels = None

    data_mode = 'normal'  #normal poison global_label
    # for Data_Mode.SINGLE_CLASS
    single_class = 1
    # for Data_Mode.POISON
    poison_fraction = 1
    poison_subject_labels = [[1],[3],[5]]
    poison_object_label = [0,2,3]
    poison_cover_labels = [[11,12],[13,14],[15,16]]
    poison_pattern_file = None # None for adaptive solid_rd pattern

    load_mode = 'normal'  #normal bottom last_affine bottom_affine all
    # backbone_model_path = None
    # backbone_model_path = home_dir+'data/benchmark_models/poisoned_bb'
    backbone_model_path = home_dir+'data/gtsrb_models/benign_all'
    #load_mode = Load_Mode.BOTTOM_AFFINE
    #for Load_Mode.ALL
    all_file = home_dir+'data/gtsrb_models/poisoned_solid_rd_2'
    #all_file = home_dir+'data/checkpoint/model.ckpt-13342'
    #for Load_Mode.BOTTOM_AFFINE
    bottom_file = home_dir+'data/gtsrb_models/poisoned_solid_rd_2'
    affine_files = [home_dir+'data/gtsrb_models/poisoned_solid_rd_2']
    affine_classes = [43]

    fix_level = 'none' #none bottom last_affine bottom_affine

    tower_name = 'tower'

    optimizer = 'sgd'
    # optimizer = 'momentum'
    base_lr = 0.1
    weight_decay = 0.00004
    #weight_decay = None



