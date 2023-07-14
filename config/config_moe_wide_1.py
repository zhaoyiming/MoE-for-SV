class Config(object):
    # system sets
    USE_CUDA=True
    RESUME=False

    #  Path
    source_path="/apdcephfs/private_maximuszhao/moe_new"
    data_path= source_path + "/data"
    save_dir = source_path + '/checkpoints/resnet56_moe_wide_all_1'

    path_checkpoint = save_dir+"/model.th"  # 断点路径
    # basenet_checkpoint = save_dir+"/model_basenet.th"  # 断点路径
    # moe_checkpoint = save_dir+"/model_moe.th"  # 断点路径



    # Model parameter
    embedding_size =128
    w_base=32
    num_classes = 10

    dataset='cifar10'
    env = 'default'
    backbone = 'resnet56'
    classify = 'softmax'
 


    train_batch_size = 1	  # batch size
    test_batch_size = 1

    optimizer = 'sgd'

    num_workers = 8  # how many workers for loading data
    print_freq = 100
    save_every = 20

    para_miu=1
    para_lambda=1

    max_epoch = 270
    lr = 1e-1  # initial learning rate
    weight_decay = 5e-4
