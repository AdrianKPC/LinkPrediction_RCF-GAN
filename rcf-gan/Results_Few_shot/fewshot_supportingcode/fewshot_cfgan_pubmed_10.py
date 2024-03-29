import time
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler
from fewshot_loss import CFLossFunc
from fewshot_network_module import SampleNet, AdvNet, InnerProductDecoder
from fewshot_utility_pubmed_10 import prepare_dir, get_input_white_noise, get_graph_target, record_as_img, avg_record, \
    tensorboard_img_writer, mask_test_edges, preprocess_graph, get_roc_score, normalise_adj
from progress.bar import Bar as Bar
from torch.utils.tensorboard import SummaryWriter


class Model:

    def __init__(self, model_label,
                 target_source, target_size, target_batch_size,
                 adversarial_training_param, generator_training_param,
                 loss_type, loss_alpha, loss_beta, threshold, loss_reg,
                 epoch):

        # init dir
        self.model_path, self.model_trace_path, self.result_path, \
        self.test_result_dir, self.mid_result_path, self.model_file = prepare_dir(model_label)

        # white noise -> x
        self.white_noise_dim = generator_training_param['input_noise_dim']
        self.white_noise_var = generator_training_param['input_noise_var']

        # target
        self.target_source = target_source
        adj, self.features = get_graph_target(target_source)
        self.n_nodes, self.feature_dim = self.features.shape
        # store original adjacency matrix (without diagonal entries) for later use
        adj_orig = adj
        self.adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        self.adj_orig.eliminate_zeros()
        self.adj_train, self.train_edges, self.val_edges, self.val_edges_false, \
        self.test_edges, self.test_edges_false = mask_test_edges(adj)

        self.net_type = generator_training_param['net_type']
        if generator_training_param['net_type'] not in ['gnn']:
            raise SystemExit('Error: The generator is only supported for dcgan, adv-dcgan and resnet structures. '
                             'Unknown source for target: {0}'.format(generator_training_param['net_type']))
        if adversarial_training_param['net_type'] not in ['gnn']:
            raise SystemExit('Error: The critic is only supported for dcgan and resnet structures. '
                             'Unknown source for target: {0}'.format(generator_training_param['net_type']))

        # Generator Net
        self.sample_net = SampleNet(self.white_noise_dim, self.feature_dim, self.n_nodes, generator_training_param)
        # Simple decoder net
        self.prod_operation = InnerProductDecoder(0.)
        # AdvNet
        self.adversarial_net = AdvNet(self.feature_dim, self.white_noise_dim, adversarial_training_param)

        # loss function: CFLossFun
        self.loss_fun = CFLossFunc(loss_type, loss_alpha, loss_beta, threshold)

        # optimization for generator
        self.optimizer_gen = optim.Adam(self.sample_net.parameters(), lr=generator_training_param['lr'],
                                        betas=(0.5, 0.999), weight_decay=generator_training_param['weight_decay'])
        self.lr_decay_gen = False
        if generator_training_param['lr_step_size_decay'] > 0:
            self.lr_decay_gen = True
            self.lr_scheduler_gen = lr_scheduler.StepLR(self.optimizer_gen,
                                                        step_size=generator_training_param['lr_step_size_decay'],
                                                        gamma=generator_training_param['lr_decay_gamma'])

        # optimization for adversarial
        self.optimizer_adv = optim.Adam(self.adversarial_net.parameters(), lr=adversarial_training_param['lr'],
                                        betas=(0.5, 0.999), weight_decay=adversarial_training_param['weight_decay'])
        self.lr_decay_adv = False
        if adversarial_training_param['lr_step_size_decay'] > 0:
            self.lr_decay_adv = True
            self.lr_scheduler_adv = lr_scheduler.StepLR(self.optimizer_adv,
                                                        step_size=adversarial_training_param['lr_step_size_decay'],
                                                        gamma=adversarial_training_param['lr_decay_gamma'])

        self.epoch = epoch
        self.lamda = loss_reg
        self.inner_ite_gen = generator_training_param['inner_ite_per_batch']
        self.inner_ite_adv = adversarial_training_param['inner_ite_per_batch']
        # tensorboard
        self.writer = SummaryWriter('runs/' + model_label)

    def train(self, load):
        # cuda setting and load previous model
        device = 'cpu'
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        if load:
            cp = torch.load(self.model_path, map_location=device)
            start_epoch = cp['epoch']
            self.sample_net.load_state_dict(cp['gen_model_state_dict'])
            self.sample_net.to(device)
            self.adversarial_net.load_state_dict(cp['adv_model_state_dict'])
            self.adversarial_net.to(device)
            self.optimizer_adv.load_state_dict(cp['adv_optimizer_state_dict'])
            self.optimizer_gen.load_state_dict(cp['gen_optimizer_state_dict'])
        else:
            start_epoch = 0
            # initilize generator
            self.sample_net.to(device)
            # self.sample_net.apply(weights_init_resnet)

            # initilize adv
            self.adversarial_net.to(device)
            # self.adversarial_net.apply(weights_init_resnet)

        # train
        # start = time.time()
        # x, loss = None, None
        adj_norm = preprocess_graph(self.adj_train).to(device)
        adj_norm = adj_norm.to_dense()
        self.features = self.features.to(device)
        self.adversarial_net.train()
        self.sample_net.train()
        # initislise bar progress plot
        bar = Bar('Training', max=self.epoch)
        
        # Array to store training/testing accuracy for graph plot outside loop
        training_roc = []
        test_roc = []
        training_ap = []
        test_ap = []

        for i in range(start_epoch, self.epoch):
            lam_d = self.lamda[i]
            # mini batch
            for i_adv in range(self.inner_ite_adv):
                # train t_nets
                self.optimizer_adv.zero_grad()
                white_noise = get_input_white_noise(self.white_noise_dim, self.white_noise_var,
                                                    self.n_nodes).detach().to(device)
                # spherical distribution
                white_noise = nn.functional.normalize(white_noise, p=2, dim=1)
                # forward again
                with torch.no_grad():
                    est_features = self.sample_net(white_noise)
                    est_adj_norm = normalise_adj(self.prod_operation(white_noise))
                est_emb = self.adversarial_net(est_features, est_adj_norm)
                target_emb = self.adversarial_net(self.features, adj_norm)
                t_batch = self.adversarial_net.net_t()
                latent_ae_loss = self.loss_fun(t_batch, white_noise, target_emb)
                latent_ae_loss_z = nn.functional.mse_loss(est_emb, white_noise)
                negloss = latent_ae_loss - self.loss_fun(t_batch, white_noise, est_emb) \
                          + lam_d * latent_ae_loss_z
                negloss.backward()
                self.optimizer_adv.step()

            # training generator/sampler
            for i_gen in range(self.inner_ite_gen):
                # train gan loss
                self.optimizer_gen.zero_grad()
                white_noise = get_input_white_noise(self.white_noise_dim, self.white_noise_var,
                                                    self.n_nodes).detach().to(device)
                # spherical distribution
                white_noise = nn.functional.normalize(white_noise, p=2, dim=1)
                with torch.no_grad():
                    t_batch = self.adversarial_net.net_t()
                    target_emb = self.adversarial_net(self.features, adj_norm)
                    est_adj_norm = normalise_adj(self.prod_operation(white_noise))
                # forward
                est_features = self.sample_net(white_noise)
                est_emb = self.adversarial_net(est_features, est_adj_norm)
                target_rec_features = self.sample_net(target_emb.detach())
                # compute loss amd backward
                observation_ae_loss = nn.functional.l1_loss(target_rec_features, self.features)
#                training_loss.append(observation_ae_loss)
                g_loss = self.loss_fun(t_batch.detach(), est_emb, target_emb.detach())
                loss = g_loss
                loss.backward()
                # Update
                self.optimizer_gen.step()

            # Validation
            hidden_emb = nn.functional.normalize(target_emb.detach(), p=2, dim=1)
            hidden_emb = hidden_emb.to(torch.device('cpu'))
            # adj_for_val = target_est_adj.to(torch.device('cpu')).detach()
            roc_curr, ap_curr = get_roc_score(hidden_emb, self.adj_orig, self.val_edges, self.val_edges_false)
            
            
            
            # plot progress
            bar.suffix = 'Epoc:{ep:.1f}|Time:{total:}|ETA:{eta:}' \
                         ' ROC:{roc_curr:.2f} | AP:{ap_curr:.2f} |'.format(
                ep=i,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                roc_curr=roc_curr,
                ap_curr=ap_curr
            )
            bar.next()
            
            #print('Training ROC score: ' + str(roc_curr))
            #print('Training AP score: ' + str(ap_curr))
            
            # Appending to training array
            training_roc.append(roc_curr)
            training_ap.append(ap_curr)

            
            hidden_emb = nn.functional.normalize(target_emb.detach(), p=2, dim=1)
            hidden_emb = hidden_emb.to(torch.device('cpu'))
            # adj_for_val = target_est_adj.to(torch.device('cpu')).detach()
            roc_score, ap_score = get_roc_score(hidden_emb, self.adj_orig, self.test_edges, self.test_edges_false)
            
            #print('Test ROC score: ' + str(roc_score))
            #print('Test AP score: ' + str(ap_score))
            
            # Appending to test array
            test_roc.append(roc_score)
            test_ap.append(ap_score)


#            plt.plot(i, roc_curr, '--',  label="Training ROC")
#            plt.plot(i, roc_score, label="Test ROC")
        


            # update scheduler information
            if self.lr_decay_gen:
                self.lr_scheduler_gen.step()
            if self.lr_decay_adv:
                self.lr_scheduler_adv.step()

            self.writer.add_scalars('validation', {'ROC': roc_curr, 'AP': ap_curr}, i + 1)


        bar.finish()
        # # save model and plot loss function
        torch.save({
            'adv_model_state_dict': self.adversarial_net.state_dict(),
            'gen_model_state_dict': self.sample_net.state_dict(),
        }, self.model_path)

#        print('Validation now ------------')
                
        #output training and testing results to file
        np.savetxt('fewshot_pubmed_10percent_.txt', np.column_stack((training_roc, training_ap, test_roc, test_ap)))




        

#        # Validation
#        hidden_emb = nn.functional.normalize(target_emb.detach(), p=2, dim=1)
#        hidden_emb = hidden_emb.to(torch.device('cpu'))
#        # adj_for_val = target_est_adj.to(torch.device('cpu')).detach()
#        roc_score, ap_score = get_roc_score(hidden_emb, self.adj_orig, self.test_edges, self.test_edges_false)

#        print('Test ROC score: ' + str(roc_score))
#        print('Test AP score: ' + str(ap_score))

#        return roc_score, ap_score

