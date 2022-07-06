import numpy as np
import torch, math
from .evaluate import accuracy
from net import Network
import dgl
import logging
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import torch
import torch.nn as nn
import torch.nn.functional as F

class Combiner:
    def __init__(self, cfg, device, num_class_list=None):
        self.cfg = cfg
        self.type = cfg['train']['combiner']['type']
        self.device = device
        self.num_class_list = torch.FloatTensor(num_class_list)
        self.epoch_number = cfg['train']['max_epoch']
        self.func = torch.nn.Sigmoid() \
            if cfg['loss']['type'] in ['FocalLoss', 'ClassBalanceFocal'] else \
            torch.nn.Softmax(dim=1)
        self.initilize_all_parameters()

        if self.cfg['train']['combiner']['type'] == 'FDS':
            self.start_smooth = self.cfg['train']['combiner']['fds']['start_smooth']

            drug_dim = self.cfg['backbone']['deeppurpose']['hidden_dim_drug'] \
                if self.cfg['dataset']['drug_encoding'] != 'Transformer' else self.cfg['backbone']['deeppurpose']['transformer_emb_size_drug']
            protein_dim = self.cfg['backbone']['deeppurpose']['hidden_dim_protein'] \
                if self.cfg['dataset']['protein_encoding'] != 'Transformer' else self.cfg['backbone']['deeppurpose']['transformer_emb_size_target']

            if self.cfg['dataset']['tier2_task'] == 'DTI':
                feature_dim = drug_dim + protein_dim
            elif self.cfg['dataset']['tier2_task'] == 'DDI':
                feature_dim = drug_dim * 2
            else:
                feature_dim = drug_dim

            self.fds = FDS(
                feature_dim=feature_dim,
                bucket_num=self.cfg['train']['combiner']['fds']['bucket_num'],
                bucket_start=self.cfg['train']['combiner']['fds']['bucket_start'],
                start_update=self.cfg['train']['combiner']['fds']['start_update'],
                start_smooth=self.cfg['train']['combiner']['fds']['start_smooth'],
                kernel=self.cfg['train']['combiner']['fds']['kernel'],
                ks=self.cfg['train']['combiner']['fds']['ks'],
                sigma=self.cfg['train']['combiner']['fds']['sigma'],
                momentum=self.cfg['train']['combiner']['fds']['momentum'],
                device=device
            )


    def initilize_all_parameters(self):
        self.alpha = self.cfg['train']['combiner']['alpha']
        self.manifold_mix_up_location = self.cfg['train']['combiner']['manifold_mix_up']['location']
        self.remix_kappa = self.cfg['train']['combiner']['remix']['kappa']
        self.remix_tau = self.cfg['train']['combiner']['remix']['tau']
        print('_'*100)
        print('combiner type: ', self.type)
        print('alpha in combiner: ', self.alpha)
        if self.type == 'manifold_mix_up':
            if 'res32' in self.cfg['backbone']['type']:
                assert self.manifold_mix_up_location in ['layer1', 'layer2', 'layer3', 'pool', 'fc']
            else:
                assert self.manifold_mix_up_location in ['layer1', 'layer2', 'layer3', 'pool', 'fc', 'layer4']
            print('location in manifold mixup: ', self.manifold_mix_up_location)
        if self.type == 'remix':
            print('kappa in remix: ', self.remix_kappa)
            print('tau in remix: ', self.remix_tau)
        print('_'*100)

    def update(self, epoch):
        self.epoch = epoch
    
    def forward(self, model, criterion, data, label, meta, meta_data, meta_label, lds_weight=None, epoch=None, training=False, **kwargs):
        if self.type == 'default':
            return eval("self.{}".format(self.type))(
                model, criterion, data, label, meta, meta_data, meta_label, lds_weight, **kwargs
            )
        elif self.type == 'FDS':
            return eval("self.{}".format(self.type))(
                model, criterion, data, label, meta, meta_data, meta_label, epoch, training, **kwargs
            )
        elif self.type == 'bbn_mix':
            return eval("self.{}".format(self.type))(
                model, criterion, data, label, meta, meta_data, meta_label, **kwargs
            )
        else:
            return eval("self.{}".format(self.type))(
                model, criterion, data, label, meta, **kwargs
            )

    def default(self, model, criterion, data, label, meta, meta_data, meta_label, lds_weight=None, **kwargs):
        data, label = data.to(self.device), label.to(self.device)
        if meta_data is not None:
            # data_b = meta["sample_data"].to(self.device)
            # label_b = meta["sample_label"].to(self.device)
            data_b = meta_data.to(self.device)
            label_b = meta_label.to(self.device)
            data_new = []
            for (data_i, data_j) in zip(data, data_b):
                if isinstance(data_i, torch.Tensor):
                    data_new.append(torch.cat([data_i, data_j], dim=0))
                elif isinstance(data_i, dgl.DGLGraph):
                    feat_i = model([data_i], feature_flag=True)
                    feat_j = model([data_j], feature_flag=True)
                    data_new.append(torch.cat([feat_i, feat_j], dim=0))
            # data = torch.cat([data, data_b], dim=0)
            # data = torch.cat([torch.cat(data, dim=0), torch.cat(data_b, dim=0)], dim=0)
            data = data_new
            label = torch.cat([label, label_b])
        if (meta_data == None) or (not isinstance(meta_data[0], dgl.DGLGraph)):
            feature = model(data, feature_flag=True)
        else:
            feature = torch.concat(data)
        output = model(feature, head_flag=True, label=label)

        if lds_weight is not None:
            lds_weight = lds_weight.to(self.device)
            loss = criterion(output, label, lds_weight, feature=feature)
        else:
            if self.cfg['setting']['type'] == 'LT Regression':
                loss = criterion(output, label, feature=feature)
            else:
                loss = criterion(output, label.long(), feature=feature)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(label.cpu().numpy(), now_result.cpu().numpy())[0]

        return loss, now_acc, label.cpu().numpy(), now_result.cpu().numpy()

    def mix_up(self, model, criterion, data, label, meta, **kwargs):
        r"""
        References:
            Zhang et al., mixup: Beyond Empirical Risk Minimization, ICLR
        """
        l = np.random.beta(self.alpha, self.alpha)
        # idx = torch.randperm(data.size(0))
        data = data.to(self.device)

        # feat_a = model(data, feature_flag=True)
        # feat_b = model(data[idx], feature_flag=True)
        feat = model(data, feature_flag=True)
        idx = torch.randperm(feat.size(0))
        feat_a = feat
        feat_b = feat[idx]
        label_a, label_b = label, label[idx]
        mixed_feat = l * feat_a + (1 - l) * feat_b
        label_a = label_a.to(self.device)
        label_b = label_b.to(self.device)

        output = model(mixed_feat, head_flag=True)
        loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)
        now_result = torch.argmax(self.func(output), 1)
        now_acc = l * accuracy(label_a.cpu().numpy(), now_result.cpu().numpy())[0] + (1 - l) * \
                  accuracy(label_b.cpu().numpy(), now_result.cpu().numpy())[0]

        return loss, now_acc, label.cpu().numpy(), now_result.cpu().numpy()


    def manifold_mix_up(self, model, criterion, data, label, meta, **kwargs):
        r"""
        References:
            Verma et al., Manifold Mixup: Better Representations by Interpolating Hidden States, ICML 2019.

        Specially, we apply manifold mixup on only one layer in our experiments.
        The layer is assigned by param ``self.manifold_mix_up_location''
        """
        l = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(image.size(0))
        data = data.to(self.device)

        label_a, label_b = label, label[idx]
        label_a = label_a.to(self.device)
        label_b = label_b.to(self.device)

        output = model(data, index=idx, layer=self.manifold_mix_up_location, coef=l)
        loss = l * criterion(output, label_a) + (1-l) * criterion(output, label_b)
        now_result = torch.argmax(self.func(output), 1)
        now_acc = l * accuracy(label_a.cpu().numpy(), now_result.cpu().numpy())[0] + (1 - l) * \
                  accuracy(label_b.cpu().numpy(), now_result.cpu().numpy())[0]
        return loss, now_acc, label.cpu().numpy(), now_result.cpu().numpy()

    def remix(self, model, criterion, data, label, meta, **kwargs):
        r"""
        Reference:
            Chou et al. Remix: Rebalanced Mixup, ECCV 2020 workshop.

        The difference between input mixup and remix is that remix assigns lambdas of mixed labels
        according to the number of images of each class.

        Args:
            tau (float or double): a hyper-parameter
            kappa (float or double): a hyper-parameter
            See Equation (10) in original paper (https://arxiv.org/pdf/2007.03943.pdf) for more details.
        """
        assert self.num_class_list is not None, "num_class_list is required"

        l = np.random.beta(self.alpha, self.alpha)
        # idx = torch.randperm(data.size(0))
        data = data.to(self.device)

        # feat_a = model(data, feature_flag=True)
        # feat_b = model(data[idx], feature_flag=True)
        feat = model(data, feature_flag=True)
        idx = torch.randperm(feat.size(0))
        feat_a = feat
        feat_b = feat[idx]

        label_a, label_b = label, label[idx]
        mixed_feat = l * feat_a + (1 - l) * feat_b

        output = model(mixed_feat, head_flag=True)

        bins = [  0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]
        label_a = np.digitize(label_a, bins=bins) - 1
        label_b = np.digitize(label_b, bins=bins) - 1
        # label_a[label_a==11] = 10
        # label_b[label_b==11] = 10


        # bins = [ 0., 3.69455, 7.3891, 11.08365, 14.7782, 18.47275, 22.1673, 25.86185, 29.5564 ]
        # label_a = np.digitize(label_a, bins=bins) - 1
        # label_b = np.digitize(label_b, bins=bins) - 1
        # label_a[label_a==8] = 7
        # label_b[label_b==8] = 7
        label_a = torch.tensor(label_a, device=self.device)
        label_b = torch.tensor(label_b, device=self.device)

        #what remix does
        # l_list = torch.empty(data.shape[0]).fill_(l).float().to(self.device)
        l_list = torch.empty(feat.shape[0]).fill_(l).float().to(self.device)
        n_i, n_j = self.num_class_list[label_a.long()], self.num_class_list[label_b.long()]
        if l < self.remix_tau:
            l_list[n_i/n_j >= self.remix_kappa] = 0
        if 1 - l < self.remix_tau:
            l_list[(n_i*self.remix_kappa)/n_j <= 1] = 1

        label_a = label_a.to(self.device)
        label_b = label_b.to(self.device)
        loss = l_list * criterion(output, label_a) + (1 - l_list) * criterion(output, label_b)
        loss = loss.mean()
        now_result = torch.argmax(self.func(output), 1)
        now_acc = (l_list * accuracy(label_a.cpu().numpy(), now_result.cpu().numpy())[0] \
                + (1 - l_list) * accuracy(label_b.cpu().numpy(), now_result.cpu().numpy())[0]).mean()
        return loss, now_acc, label.cpu().numpy(), now_result.cpu().numpy()

    def bbn_mix(self, model, criterion, data, label, meta, meta_data, meta_label, **kwargs):
        r"""
        Reference:
            Zhou et al. BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition, CVPR 2020.

        We combine the sampling method of BBN, which consists of a uniform sampler and a reverse sampler, with input mixup.

        For more details about these two samplers, you can read the original paper https://arxiv.org/abs/1912.02413.
        """
        l = np.random.beta(self.alpha, self.alpha) # beta distribution

        data_a = data.to(self.device)
        # data_b = meta["sample_data"].to(self.device)
        data_b = meta_data.to(self.device)      

        feat_a = model(data_a, feature_flag=True)
        feat_b = model(data_b, feature_flag=True)

        # label_a, label_b = label.to(self.device), meta["sample_label"].to(self.device)
        label_a, label_b = label.to(self.device), meta_label.to(self.device)

        # mix up two features
        mixed_feat = l * feat_a + (1 - l) * feat_b

        mixed_output = model(mixed_feat, head_flag=True)

        loss = l * criterion(mixed_output, label_a) + (1 - l) * criterion(mixed_output, label_b)

        now_result = torch.argmax(self.func(mixed_output), 1)
        now_acc = (
                l * accuracy(label_a.cpu().numpy(), now_result.cpu().numpy())[0]
                + (1 - l) * accuracy(label_b.cpu().numpy(), now_result.cpu().numpy())[0]
        )
        return loss, now_acc, label.cpu().numpy(), now_result.cpu().numpy()

    def dive(self, model, criterion, data, label, meta, **kwargs):

        if not hasattr(self, 'model_t'):
            print('Loading the teacher model in DiVE')
            self.model_t = Network(model.model, self.cfg, mode="test", num_class=len(self.num_class_list))
            self.model_t.load_model(self.cfg['train']['combiner']['dive']['teacher_model'])
            self.model_t = torch.nn.DataParallel(self.model_t).cuda()
            self.model_t.eval()

        data, label = data.to(self.device), label.to(self.device)
        output_s = model(data)

        with torch.no_grad():
            output_t = self.model_t(data)

        loss = criterion(output_s, output_t, label)
        now_result = torch.argmax(self.func(output_s), 1)
        now_acc = accuracy(label.cpu().numpy(), now_result.cpu().numpy())[0]
        return loss, now_acc, label.cpu().numpy(), now_result.cpu().numpy()

    def FDS(self, model, criterion, data, label, meta, meta_data, meta_label, epoch, training, **kwargs):
        data, label = data.to(self.device), label.to(self.device)
        if meta_data is not None:
            # data_b = meta["sample_data"].to(self.device)
            # label_b = meta["sample_label"].to(self.device)
            data_b = meta_data.to(self.device)
            label_b = meta_label.to(self.device)
            data_new = []
            for (data_i, data_j) in zip(data, data_b):
                if isinstance(data_i, torch.Tensor):
                    data_new.append(torch.cat([data_i, data_j], dim=0))
                elif isinstance(data_i, dgl.DGLGraph):
                    feat_i = model([data_i], feature_flag=True)
                    feat_j = model([data_j], feature_flag=True)
                    data_new.append(torch.cat([feat_i, feat_j], dim=0))
            # data = torch.cat([data, data_b], dim=0)
            # data = torch.cat([torch.cat(data, dim=0), torch.cat(data_b, dim=0)], dim=0)
            data = data_new
            label = torch.cat([label, label_b])
        if (meta_data == None) or (not isinstance(meta_data[0], dgl.DGLGraph)):
            feature = model(data, feature_flag=True)
        else:
            feature = torch.concat(data)

        if training and epoch >= self.start_smooth:
            feature = self.fds.smooth(feature, label, epoch)

        output = model(feature, head_flag=True, label=label)

        if self.cfg['setting']['type'] == 'LT Regression':
            loss = criterion(output, label, feature=feature)
        else:
            loss = criterion(output, label.long(), feature=feature)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(label.cpu().numpy(), now_result.cpu().numpy())[0]

        return loss, now_acc, label.cpu().numpy(), now_result.cpu().numpy()


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


class FDS(nn.Module):

    def __init__(self, feature_dim, bucket_num=100, bucket_start=3, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9, device='cuda:0'):
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_window = self._get_kernel_window(kernel, ks, sigma)
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth
        self.device = device

        self.register_buffer('epoch', torch.zeros(1).fill_(start_update).to(device))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim).to(device))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim).to(device))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim).to(device))
        self.register_buffer('running_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim).to(device))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim).to(device))
        self.register_buffer('smoothed_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim).to(device))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start).to(device))

    @staticmethod
    def _get_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(map(laplace, np.arange(-half_ks, half_ks + 1)))

        print(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        return torch.tensor(kernel_window, dtype=torch.float32).cuda()

    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        self.smoothed_mean_last_epoch = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        self.smoothed_var_last_epoch = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            print(f"Updated smoothed statistics on Epoch [{epoch}]!")

    def update_running_stats(self, features, labels, epoch):
        if epoch < self.epoch:
            return

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                curr_feats = features[labels <= label]
            elif label == self.bucket_num - 1:
                curr_feats = features[labels >= label]
            else:
                curr_feats = features[labels == label]
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            self.num_samples_tracked[int(label - self.bucket_start)] += curr_num_sample
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[int(label - self.bucket_start)]))
            factor = 0 if epoch == self.start_update else factor
            self.running_mean[int(label - self.bucket_start)] = \
                (1 - factor) * curr_mean + factor * self.running_mean[int(label - self.bucket_start)]
            self.running_var[int(label - self.bucket_start)] = \
                (1 - factor) * curr_var + factor * self.running_var[int(label - self.bucket_start)]

        print(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels, epoch):
        if epoch < self.start_smooth:
            return features

        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                features[labels <= label] = calibrate_mean_var(
                    features[labels <= label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
            elif label == self.bucket_num - 1:
                features[labels >= label] = calibrate_mean_var(
                    features[labels >= label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
            else:
                features[labels == label] = calibrate_mean_var(
                    features[labels == label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
        return features