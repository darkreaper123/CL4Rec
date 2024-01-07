import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear, PairwiseDistance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator_raw_multi_task_similarity(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator_raw_multi_task_similarity, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        else:
            raise NotImplementedError


    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        # Equation (3)
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            # Equation (7) & (9)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings)           # (n_users + n_entities, out_dim)
        return embeddings


class Graph_Bi_interaction_with_UFO_SPACE(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations, device, items_id, A_in=None,
                 user_pre_embed=None, item_pre_embed=None, task_ids = None,
                 n_cf_batch = None, users_group = None, items_group = None,
                 n_clusters_users = None, n_clusters_items = None):

        super(Graph_Bi_interaction_with_UFO_SPACE, self).__init__()
        self.use_task_mask = args.use_task_mask
        if args.weight_loss is not None:
            self.weight_loss = args.weight_loss
        else:
            self.weight_loss = 1. - 1./n_cf_batch
        self.sigmoid_activation_before_similarity = args.sigmoid_activation_before_similarity
        self.use_two_loss = False
        self.epoch_binary = False
        self.task_ids = task_ids
        self.task_id = None
        self.task_ids_index = 0
        self.limit = args.limit
        self.limits = args.limits
        self.limits = [float(elem) for elem in self.limits.split(',')]
        self.use_pretrain = args.use_pretrain
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.args = args
        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.device = device
        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim, padding_idx = 0)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))
        
        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed)
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
        else:
            nn.init.xavier_uniform_(self.entity_user_embed.weight)
        if True:
            nn.init.xavier_uniform_(self.relation_embed.weight)
            nn.init.xavier_uniform_(self.trans_M)

        self.aggregator_layers = nn.ModuleList()
        sum_n_embed = self.embed_dim
        self.bottleneck_dim = 5
        #self.embedding_task_layers = nn.ModuleList()
        #self.embedding_task_layers_2 = nn.ModuleList()
        self.embedding_task_layer_users_first = nn.Embedding(len(args.task_ids.split(',')), self.embed_dim)
        self.mapping_to_users_masks = nn.ModuleList([nn.ModuleList([nn.Linear(self.embed_dim, self.bottleneck_dim), nn.Linear(self.bottleneck_dim, self.embed_dim)]) for i in range(len(self.task_ids))])
        self.embedding_task_layer_items_first = nn.Embedding(len(args.task_ids.split(',')), self.embed_dim)
        self.mapping_to_items_masks = nn.ModuleList([nn.ModuleList([nn.Linear(self.embed_dim, self.bottleneck_dim), nn.Linear(self.bottleneck_dim, self.embed_dim)]) for i in range(len(self.task_ids))])
        self.n_tasks = len(args.task_ids.split(','))
        if True:
            aggregator_layers_one = nn.ModuleList()
            for k in range(self.n_layers):
                sum_n_embed += self.conv_dim_list[k + 1]
                aggregator_layers_one.append(Aggregator_raw_multi_task_similarity(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))
        self.aggregator_layers = aggregator_layers_one
        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_users + self.n_entities, self.n_users + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False
        self.items_id = items_id
        self.n_items = items_id.shape[0]
        self.user_ids = torch.arange(self.n_users, dtype=torch.long).to(device) + self.n_entities
        self.entity_ids = torch.arange(self.n_entities - self.items_id.shape[0], dtype=torch.long).to(device) + self.items_id.shape[0]
        self.just_tunning_embedding = False
        nn.init.xavier_uniform_(self.embedding_task_layer_users_first.weight)
        nn.init.xavier_uniform_(self.embedding_task_layer_items_first.weight)
        self.model_epoch_binary_for_multi_tasks = [False for i in range(len(self.task_ids))]
        self.check_test = False
        self.just_one_task_mask_for_items = args.just_one_task_mask_for_items
        self.just_one_task_mask_for_users = args.just_one_task_mask_for_users
        self.mf = args.mf
        #if users_group is not None:
        if False:
            self.users_group = np.zeros(self.n_users)
            self.users_group[list(users_group.keys())] = np.asarray(list(users_group.values()))
            self.users_group = torch.LongTensor([list(self.users_group) for i in range(len(self.task_ids))])
            self.n_clusters_users = n_clusters_users
            self.embedding_task_layer_users = nn.ModuleList([nn.Embedding(n_clusters_users, self.embed_dim) for i in range(len(self.task_ids))])
            for i in range(len(self.task_ids)):
                nn.init.xavier_uniform_(self.embedding_task_layer_users[i].weight)
        if False:
            self.items_group = np.zeros(self.n_items)
            self.items_group[list(items_group.keys())] = np.asarray(list(items_group.values()))
            self.items_group = torch.LongTensor([list(self.items_group) for i in range(len(self.task_ids))])
            self.n_clusters_items = n_clusters_items
            self.embedding_task_layer_items = nn.ModuleList([nn.Embedding(n_clusters_items, self.embed_dim) for i in range(len(self.task_ids))])
            for i in range(len(self.task_ids)):
                nn.init.xavier_uniform_(self.embedding_task_layer_items[i].weight)
            
        #self.check_n_true_masks_each_user_each_task = np.asarray([[1 for j in range(self.n_users)] for i in range(self.n_tasks)])
    def set_weights_embedding(self):
        self.entity_user_embed.weight = nn.Parameter(torch.from_numpy(self.old_embedding).to(self.device).to(torch.float32))
    def load_task_id(self, device):
        self.task_id = torch.tensor([[self.task_ids[self.task_ids_index]]], dtype = torch.long).to(device)
        self.raw_task_id = self.task_ids[self.task_ids_index]
    def get_task_embedding_binary(self, embedding, user_or_item = 1, for_optimizing = False):
        recent_task_id_index = self.task_ids_index
        begin_index = recent_task_id_index
        if self.check_test:
            begin_index = 0
        true_task_embedding = None
        for i in range(begin_index, recent_task_id_index + 1):
            if user_or_item == 1:
                if True:
                    #task_embedding = self.embedding_task_layer_users_first(self.task_id).view(1, -1)
                    #task_embedding = (torch.ones_like(embedding) + 1.).to(self.device) * task_embedding
                    #task_embedding = torch.concat([embedding, task_embedding], dim = 1)
                    
                    if self.args.ensemble_learning and self.check_test:
                        if True:
                            if False:
                                task_embedding = torch.sigmoid(self.embedding_task_layer_users[i](self.users_group[i].to(self.device)))
                                task_embedding += torch.sigmoid(self.embedding_task_layer_users_first(self.task_id).view(1, -1))
                                task_embedding = task_embedding * 0.5
                            else:
                                task_embedding = torch.sigmoid(self.embedding_task_layer_users[i](self.users_group[i].to(self.device)))
                                task_embedding = torch.concat([torch.ones_like(task_embedding) * torch.sigmoid(self.embedding_task_layer_users_first(self.task_id).view(1, -1)), task_embedding], dim = 0).view(2, task_embedding.shape[0], task_embedding.shape[1])
                                task_embedding, _ = torch.max(task_embedding, dim = 0)
                    else:
                        if True:
                            if not self.just_one_task_mask_for_users:
                                task_embedding = torch.sigmoid(self.embedding_task_layer_users[i](self.users_group[i].to(self.device)))
                            else:
                                task_embedding = torch.sigmoid(self.embedding_task_layer_users_first(self.task_id).view(1, -1))
            else:
                if True:
                    #task_embedding = self.embedding_task_layer_items_first(self.task_id).view(1, -1)
                    #task_embedding = (torch.ones_like(embedding) + 1.).to(self.device) * task_embedding
                    #task_embedding = torch.concat([embedding, task_embedding], dim = 1)
                    if self.args.ensemble_learning and self.check_test:
                        if True:
                            if False:
                                task_embedding = torch.sigmoid(self.embedding_task_layer_items[i](self.items_group[i].to(self.device)))
                                task_embedding += torch.sigmoid(self.embedding_task_layer_items_first(self.task_id).view(1, -1))
                                task_embedding = task_embedding * 0.5
                            else:
                                task_embedding = torch.sigmoid(self.embedding_task_layer_items[i](self.items_group[i].to(self.device)))
                                task_embedding = torch.concat([torch.ones_like(task_embedding) * torch.sigmoid(self.embedding_task_layer_items_first(self.task_id).view(1, -1)), task_embedding], dim = 0).view(2, task_embedding.shape[0], task_embedding.shape[1])
                                task_embedding, _ = torch.max(task_embedding, dim = 0)
                    else:
                        if True:
                            if not self.just_one_task_mask_for_items:
                                task_embedding = torch.sigmoid(self.embedding_task_layer_items[i](self.items_group[i].to(self.device)))
                            else:
                                task_embedding = torch.sigmoid(self.embedding_task_layer_items_first(self.task_id).view(1, -1))
                            
            if self.epoch_binary:
                task_embedding_above_limit = nn.Threshold(self.limits[i], 0.)(task_embedding)
                task_embedding = nn.Threshold(-self.limits[i], 1.)(-task_embedding_above_limit)
            try:
                true_task_embedding += task_embedding
            except:
                true_task_embedding = task_embedding
        if self.check_test:
            true_task_embedding = -nn.Threshold(-1., -1.)(-true_task_embedding)
        return true_task_embedding
    def get_task_embedding_similarity(self, weights, limits):
        new_weights = []
        for index, weight in enumerate(weights):
            weight = nn.Threshold(limits[index], 0.)(weight)
            weight = nn.Threshold(-limits[index], 1.)(-weight)
            new_weights.append(weight.view(1, -1))
        new_weights = torch.concat(new_weights, dim = 0)
        return new_weights
    def count_weight_similarity(self, index, weight):
        #print(torch.sum(weight, dim = -1))
        #print(index)
        norm_weight = []
        if True:
            with torch.no_grad():
                past_weight = weight[self.task_ids[:index]]
                if self.sigmoid_activation_before_similarity:
                    past_weight = torch.sigmoid(past_weight)
                    past_weight = self.get_task_embedding_similarity(past_weight, list(np.asarray(self.limits)[self.task_ids[:index]]))
            recent_weight = weight[self.task_ids[index:index + 1]]
            if self.sigmoid_activation_before_similarity:
                recent_weight = torch.sigmoid(recent_weight)
            similarity = torch.matmul(recent_weight, past_weight.transpose(1, 0))
            #print(weight)
            for i in range(recent_weight.shape[0]):
                for j in range(past_weight.shape[0]):
                    norm_weight.append(torch.sqrt((recent_weight[i:i+1]**2).sum(dim = -1) * (past_weight[j:j+1]**2).sum(dim = -1)))
            norm_weight = torch.concat(norm_weight, dim = 0)    
            norm_weight = norm_weight.reshape(similarity.shape[0], similarity.shape[1])
        #print(similarity/norm_weight)
        return similarity/norm_weight
    def constrative_loss_task_mask(self):
        index = self.task_ids_index
        if index == 0 or not self.use_two_loss:
            return 0
        
        weight_similarity = self.count_weight_similarity(index, self.embedding_task_layer.weight)
        return weight_similarity.mean(dim = -1).mean(dim = -1)
    def feed_raw_embedding_via_task_masks(self):
        ego_embed = self.entity_user_embed.weight
        #drop
        if self.use_task_mask: 
            users_ego_embed = ego_embed[self.n_entities:]
            items_ego_embed = ego_embed[:self.n_entities]
            users_task_embedding = self.get_task_embedding_binary(users_ego_embed, 1)
            users_ego_embed = users_ego_embed * users_task_embedding
            items_task_embedding = self.get_task_embedding_binary(items_ego_embed, 0)
            items_ego_embed = items_ego_embed * items_task_embedding
            new_ego_embed = torch.concat([items_ego_embed, users_ego_embed])
        else:
            new_ego_embed = ego_embed
        return new_ego_embed
    def calc_cf_embeddings(self):
        new_ego_embed = self.feed_raw_embedding_via_task_masks()
        all_embed = [new_ego_embed]
        if not self.mf:
            if self.just_tunning_embedding:
                with torch.no_grad():
                    for idx, layer in enumerate(self.aggregator_layers):
                        new_ego_embed = layer(new_ego_embed, self.A_in)
                        norm_embed = F.normalize(new_ego_embed, p=2, dim=1)
                        all_embed.append(norm_embed)
            else:
                if True:
                    for idx, layer in enumerate(self.aggregator_layers):
                        new_ego_embed = layer(new_ego_embed, self.A_in)
                        norm_embed = F.normalize(new_ego_embed, p=2, dim=1)
                        all_embed.append(norm_embed)

            # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)       # (n_users + n_entities, concat_dim)
        return all_embed


    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        all_embed = self.calc_cf_embeddings()                       # (n_users + n_entities, concat_dim)
        user_embed = all_embed[user_ids]                            # (cf_batch_size, concat_dim)
        item_pos_embed = all_embed[item_pos_ids]                    # (cf_batch_size, concat_dim)
        item_neg_embed = all_embed[item_neg_ids]                    # (cf_batch_size, concat_dim)

        # Equation (12)













        
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)   # (cf_batch_size)

        # Equation (13)
        # cf_loss = F.softplus(neg_score - pos_score)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss
    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        #h_embed = self.entity_user_embed.weight[h_list] * self.get_task_embedding_binary_attention_score()
        #t_embed = self.entity_user_embed.weight[t_list] * self.get_task_embedding_binary_attention_score()
        if True:
          new_ego_embed = self.feed_raw_embedding_via_task_masks()
          h_embed = new_ego_embed[h_list]
          t_embed = new_ego_embed[t_list]
        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)


    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        all_embed = self.calc_cf_embeddings()           # (n_users + n_entities, concat_dim)
        user_embed = all_embed[user_ids]                # (n_users, concat_dim)
        item_embed = all_embed[item_ids]                # (n_items, concat_dim)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_users, n_items)
        return cf_score
    def calc_full_loss(self, user_ids, item_pos_ids, item_neg_ids):
        loss_cf = self.calc_cf_loss(user_ids, item_pos_ids, item_neg_ids)
        loss_task_mask_similary = self.constrative_loss_task_mask()
        return loss_cf, loss_task_mask_similary, self.weight_loss * loss_cf - (1 - self.weight_loss) * loss_task_mask_similary

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.calc_score(*input)
        if mode == 'train_full_loss':
            return self.calc_full_loss(*input)


