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
    batch_size= 32 # 128 for gtsrb
    num_epochs = 90
    num_gpus = 2

    shuffle = True

    crop_size = 300
    mean = 127.5

    home_dir = '/home/tangdi/'
    log_dir = home_dir+'logs/'
    data_dir = home_dir+'data/imagenet/'

    checkpoint_folder = home_dir+'data/checkpoint/'


    net_mode = 'normal'
    # for Net_Mode.BACKDOOR_DEF
    loss_lambda = 0.1 # for gtsrb
    #loss_lambda = 0.01 # for megaface
    # for Net_Mode.TRIPLE_LOSS
    use_triplet_loss = False
    size_triplet_slice = 4
    num_slices_one_batch = batch_size // size_triplet_slice

    build_level = 'logits'

    selected_labels = None

    data_mode = 'normal'
    # for Data_Mode.SINGLE_CLASS
    single_class = 1
    # for Data_Mode.POISON
    poison_fraction = 1
    poison_subject_labels = [[2]]
    poison_object_label = [0]
    poison_cover_labels = [[11,12,13]]
    poison_pattern_file = None # None for adaptive solid_rd pattern

    load_mode = 'normal'
    #load_mode = Load_Mode.BOTTOM_AFFINE
    #for Load_Mode.ALL
    all_file = home_dir+'data/gtsrb_models/poisoned_solid_rd_2'
    #all_file = home_dir+'data/checkpoint/model.ckpt-13342'
    #for Load_Mode.BOTTOM_AFFINE
    bottom_file = home_dir+'data/gtsrb_models/poisoned_solid_rd_2'
    affine_files = [home_dir+'data/gtsrb_models/poisoned_solid_rd_2']
    affine_classes = [43]

    fix_level = 'none'

    num_examples_per_epoch = 0
    num_classes = 1

    tower_name = 'tower'

    optimizer = 'momentum'
    base_lr = 0.1
    weight_decay = 0.00004
    #weight_decay = None



