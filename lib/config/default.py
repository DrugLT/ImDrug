import torch
import torch.nn.functional as F

config=dict(
    dataset=dict(
        drug_encoding='DGL_GCN', 
        # choose from ['Morgan', 'Pubchem', 'Daylight', 'rdkit_2d_normalized', 'ESPF', 'CNN', 'CNN_RNN', 'Transformer', 'MPNN', 'ErG', 'DGL_GCN', 'DGL_NeuralFP', 'DGL_AttentiveFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred'], 
        # check DeepPurpose.utils.encode_drug
        protein_encoding='Transformer', 
        # choose from ['AAC', 'PseudoAAC', 'Conjoint_triad', 'Quasi-seq', 'ESPF', 'CNN', 'CNN_RNN', 'Transformer'],
        # check DeepPurpose.utils.encode_protein
        tier1_task='single_pred', # choose from ['single_pred', 'multi_pred', 'generation']
        tier2_task='ADME', 
        # single_pred: ['ADME', 'Tox', 'QM', ’BioAct', 'Yields', 'Catalyst', 'ReactType'], 
        # multi_pred: ['DTI', 'DDI', 'PPI', 'Catalyst', 'ReactType']

        dataset_name='BBB_Martins', # choose from ['BBB_Martins', 'Tox21', 'HIV', 'QM9', 'SBAP', 'DrugBank', 'USPTO_50k', 'USPTO-1k-TPL', 'USPTO_500_MT', 'USPTO-Yields', 'USPTO-Catalyst']

        # path='/apdcephfs/share_1364275/shenyuan/DrugLT/data',
        # path='/lanqingli/code/ImDrug/data',
        path='./data',
        split=dict(
            method='random', # choose from ['standard', 'random', 'scaffold', 'time', 'combination', 'group', 'cold_{ENTITY1_NAME}', 'open-random', 'open-scaffold', 'open-time', 'open-combination', 'open-group', 'open-cold_{ENTITY1_NAME}']
            scale=None,
            frac=[0.7, 0.1, 0.2],
            lt_frac=[0.3, 0.4, 0.3],
            open_frac=0.2,
            by_class=True,
            label_name='Y', # for by_class=True
            label_weight_name='Y_Weight', # class weights (1/n_samples for by_class=True, 1 for by_class=False by default)
            lt_label_name='LT_Class', # long-tailed classes
            time_column='Year', # for time split
            column_name=['DrugID', 'Target_ID']
        ),
    ),
    backbone=dict(
        # drug_type='MLP', # choose from ['MLP', 'GCN', 'GIN', 'BERT']
        drug_pretrained_model='',
        # protein_type='MLP', # choose from ['MLP', 'GCN', 'GIN', 'BERT']
        protein_pretrained_model='',
        freeze=False, 
        pretrain=False,
        # adapted from https://github.com/kexinhuang12345/DeepPurpose/blob/4e74421104b854aa241b9d7d12f4a3cbd51d034f/DeepPurpose/utils.py

        deeppurpose=dict(
            input_dim_drug = 1024, 
            input_dim_protein = 4096,
            hidden_dim_drug = 256, 
            hidden_dim_protein = 256,
            cls_hidden_dims = [1024, 1024, 512],
            mlp_hidden_dims_drug = [1024, 256, 64],
            mlp_hidden_dims_target = [1024, 256, 64],
            transformer_emb_size_drug = 128,
            transformer_intermediate_size_drug = 512,
            transformer_num_attention_heads_drug = 8,
            transformer_n_layer_drug = 8,
            transformer_emb_size_target = 64,
            transformer_intermediate_size_target = 256,
            transformer_num_attention_heads_target = 4,
            transformer_n_layer_target = 2,
            transformer_dropout_rate = 0.1,
            transformer_attention_probs_dropout = 0.1,
            transformer_hidden_dropout_rate = 0.1,
            mpnn_hidden_size = 50,
            mpnn_depth = 3,
            cnn_drug_filters = [32,64,96],
            cnn_drug_kernels = [4,6,8],
            cnn_target_filters = [32,64,96],
            cnn_target_kernels = [4,8,12],
            rnn_Use_GRU_LSTM_drug = 'GRU',
            rnn_drug_hid_dim = 64,
            rnn_drug_n_layers = 2,
            rnn_drug_bidirectional = True,
            rnn_Use_GRU_LSTM_target = 'GRU',
            rnn_target_hid_dim = 64,
            rnn_target_n_layers = 2,
            rnn_target_bidirectional = True,
            gnn_hid_dim_drug = 64,
            gnn_num_layers = 3,
            # gnn_activation = F.relu,
            neuralfp_max_degree = 10,
            neuralfp_predictor_hid_dim = 128,
            # neuralfp_predictor_activation = torch.tanh,
            attentivefp_num_timesteps = 2
        ),
    ),
    neck=dict(
        type='Concat' # choose from ['GAP', 'Identity', 'SiLU', 'Concat']
    ),
    head=dict(
        type='MLP', # choose from  ['FCNorm', 'LWS', 'MLP']  
        hidden_dims_lst=[256, 256, 256],
        bias=True,
    ),
    network=dict(
        pretrained=False,
        pretrained_model='',
    ),
    loss=dict(
        type='CrossEntropy', # choose from ['CrossEntropy', 'BalancedSoftmaxCE', 'ClassBalanceCE', 'ClassBalanceFocal', 'CDT', 'CostSensitiveCE', 'CrossEntropyLabelSmooth', 'CrossEntropyLabelAwareSmooth', 'SEQL', 'FocalLoss', 'InfluenceBalancedLoss', 'LDAMLoss', 'DiVEKLD', 'MSE']
        ClassBalancedCE=dict(
            BETA=0.999,
        ),
        CostSensitiveCE=dict(
            GAMMA=1.0,
        ),
        ClassBalanceFocal=dict(
            BETA=0.999,
            GAMMA=0.5,
        ),
        CrossEntropyLabelSmooth=dict(
            EPSILON=0.1,
        ),
        CrossEntropyLabelAwareSmooth=dict(
            smooth_head=0.4,
            smooth_tail=0.1,
            shape='concave',
        ),
        FocalLoss=dict(
            GAMMA=2.0, 
        ),
        LDAMLoss=dict(
            scale=30.0,
            max_margin=0.5,
        ),
        CDT=dict(
            GAMMA=0.2,
        ),
        SEQL=dict(
            GAMMA=0.9,
            LAMBDA=0.005,
        ),
        InfluenceBalancedLoss=dict(
            ALPHA=1000.,
        ),
        DiVEKLD=dict(
            power_norm=False,
            power=0.5,
            temperature=2.0,
            ALPHA=0.5,
            baseloss='ClassBalanceFocal',
        ),
        FocalR=dict(
            BETA=0.2,
            GAMMA=1,
            choice='l1',
        ),
        BalancedMSELoss=dict(
            SIGMA=8.0,
        ),
        LDS=dict(
            base_loss='mse',  # choose from ['mse', 'l1']
            reweight='sqrt_inv',  # choose from ['sqrt_inv', 'inverse']
            kernel='triang',  # choose from ['gaussian', 'triang', 'laplace']
            SIGMA=2,
            ks=5
        ),
    ),


    train=dict(
        batch_size=128,
        max_epoch=200,
        auto_resume=False,
        distributed=False,
        num_workers=8,
        shuffle=True,
        local_rank=0, # local rank for distributed training
        combiner=dict(
            type='default',
            alpha=1.0,
            manifold_mix_up=dict(
                location='pool'
            ),
            remix=dict(
                kappa=3.0,
                tau=0.5
            ),
            dive=dict(
                teacher_model=''
            ),
            fds=dict(
                bucket_num=100,
                bucket_start=0,
                start_update=0,
                start_smooth=1,
                kernel='triang',  # choose from ['gaussian', 'triang', 'laplace']
                ks=9,
                sigma=1.0,
                momentum=0.9
            )

        ),
        sampler=dict(
            type='default',
            weighted_sampler=dict(
                type='balance',
            ),
            bbn_sampler=dict(
                type='reverse',
            ),
        ),
        optimizer=dict(
            type='ADAM', # choose from ['SGD', 'ADAM']
            lr=1e-3, # learning rate
            momentum=0.9,
            wc=2e-4, # weight decay
        ),
        lr_scheduler=dict(
            # type='multistep', # choose from ['multistep', 'cosine', 'warmup']
            # lr_step=[40, 50],
            # lr_factor=0.1,
            # warmup_epoch=5,
            # cosine_decay_end=0,
            type='warmup', 
            lr_step=[120, 160],
            lr_factor=0.01,
            warmup_epoch=20,
            cosine_decay_end=0,
        ),
        two_stage=dict(
            drw=False,
            drs=False,
            start_epoch=1,
        ),
        tensorboard=dict(
            enable=True,
        )

    ),
    test=dict(
        batch_size=64,
        exp_id='',
        model_file='best_model.pth',
        num_workers=8,
    ),
    setting=dict(
        type='LT Classification', # choose from ['Imbalanced Learning', 'LT Classification', 'LT Regression', 'Open LT']
        num_class=10, # only effective for LT Regression
    ),
    eval_mode=False,
    # output_dir='/apdcephfs/private_lianglzeng/DrugLT/output',
    # output_dir='/apdcephfs/share_1364275/shenyuan/DrugLT/output',
    # output_dir='/lanqingli/code/ImDrug/output',
    output_dir='./output',
    name='default',
    seed=2,
    use_gpu=True,
    gpu_id=0,
    resume_model='',
    resume_mode='all',
    save_step=5,
    show_step=5,
    valid_step=5,
    pin_memory=True,
    debug=False
)