import json
import pickle
import sys
from typing import Optional, Callable, List
import os.path as osp
import torch
from torch_geometric.data import (InMemoryDataset, Data)


class FB15K237_Dataset(InMemoryDataset):

    def __init__(self, root: str, relation_id: int,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        assert relation_id in range(237)
        self.relation_id = relation_id
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_dataset_dir(self) -> str:
        return osp.join(self.raw_dir, '../../fb15k237')

    @property
    def num_relations(self) -> str:
        return self.data.edge_type.max().item() + 1

    @property
    def num_classes(self) -> str:
        return self.data.train_y.max().item() + 1

    @property
    def raw_file_names(self) -> List[str]:
        files = [
            f'train_entity.txt',
            'test_entity.txt',
            'valid_entity.txt',
        ]
        return files

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
        

    def get_raw_relation_name(self):
        relation_dict = dict()
        with open(self.raw_dataset_dir + '/relation2id.txt', 'r') as f:
            f.readline()
            g = f.readlines()
            for string in g:
                relation, id = string.replace('\n', '').split("\t")
                relation_dict[int(id)] = relation
        return relation_dict[self.relation_id]


    def download(self):
        # print("Enter FB15K237 Download function")
        return


    def entity_type_split(self):

        with open(self.raw_dataset_dir + "/relation_type.pickle", 'rb') as f:
            relation_type = pickle.load(f)
            classes = relation_type[self.relation_id]

        search = ['train', 'test', 'valid']
        for name in search:
            with open(self.raw_dataset_dir + '/' + name + '.txt', 'r') as f:
                f.readline()
                globals()[name + "_graph"] = f.readlines()

        total_dict = dict()
        node_set = set()
        for name in search:
            globals()[name + "_node"] = []

        for name in search:
            for triple_str in globals()[name + '_graph']:
                triple = triple_str.replace('\n', '').split("\t")

                if triple[0] not in total_dict.keys() and triple[1] == self.raw_relation_name and triple[2] in classes:
                    total_dict[triple[0]] = triple[2]
                    index = classes.index(triple[2])
                    globals()[name + '_node'].append(triple[0])

                node_set.add(triple[0])
                node_set.add(triple[2])

        for node in node_set:
            if node not in total_dict.keys():
                total_dict[node] = 'other'

        total_json = json.dumps(total_dict)
        with open(self.raw_dir + '/' + self.relation_name + '.json', "w") as w:
            json.dump(total_dict, w)

        for name in search:
            with open(self.raw_dir + "/" + name + "_node.txt", "w") as w:
                for node in globals()[name + "_node"]:
                    w.write(node + "\n")


    def read_train_and_test(self, node_list):
        node_list_train = list()
        node_list_test = list()

        search = ['train', 'test']
        for name in search:
            with open(self.raw_dir + "/" + name + "_node.txt", 'r') as f:
                node_read = f.readlines()
                for node in node_read:
                    node = node.replace('\n', '')

                    if name == 'train':
                        node_list_train.append(node)
                    else:
                        node_list_test.append(node)
        
        if len(node_list_test) == 0:
            print("Relation "  + str(self.relation_id) + " is not exist in test set")
            sys.exit(0)

        node_list_other = list(set(node_list) - set(node_list_train) - set(node_list_test))
        return node_list_train, node_list_test, node_list_other


    def process(self):

        self.raw_relation_name = self.get_raw_relation_name()
        self.relation_name = self.raw_relation_name.split("/")[-1]
        self.entity_type_split()

        with open(self.raw_dir + "/" + self.relation_name + '.json', 'r') as obj:
            node_type = json.load(obj)

        graph = []
        node_set = set()
        edge_set = set()
        with open(self.raw_dataset_dir + '/train.txt', 'r') as f:
            f.readline()
            g = f.readlines()
            for triple_str in g:
                triple = triple_str.replace('\n', '').split("\t")

                if triple[1] != self.raw_relation_name:
                    node_set.add(triple[0])
                    node_set.add(triple[2])
                    edge_set.add(triple[1])
                    graph.append(triple)

        node_list = list(node_set)
        edge_list = list(edge_set)

        edge_dict = dict()
        node_dict = dict()

        with open(self.raw_dataset_dir + '/relation2id.txt', 'r') as f:
            f.readline()
            g = f.readlines()
            for string in g:
                relation, id = string.replace('\n', '').split("\t")
                edge_dict[relation] = int(id)

            relation_num = len(edge_dict)
            for string in g:
                relation, id = string.replace('\n', '').split("\t")
                edge_dict["reverse_" + relation] = edge_dict[relation] + relation_num

        with open(self.raw_dataset_dir + '/entity2id.txt', 'r') as f:
            f.readline()
            g = f.readlines()
            for string in g:
                entity, id = string.replace('\n', '').split("\t")
                node_dict[entity] = int(id)

        class_dict = {}

        class_set = set()
        for key in node_type:
            class_set.add(node_type[key])
        for i, class_name in enumerate(class_set):
            class_dict[class_name] = i

        node_list_train, node_list_test, node_list_other = self.read_train_and_test(node_list)

        with open(self.processed_dir + '/id_entity_train', "w") as w:
            w.write("id      entity     class\n")
            for entity in node_list_train:
                w.write(str(node_dict[entity]) + "  " + entity + "  " + node_type[entity] + "\n")

        with open(self.processed_dir + '/id_entity_test', "w") as w:
            w.write("id      entity     class\n")
            for entity in node_list_test:
                w.write(str(node_dict[entity]) + "  " + entity + "  " + node_type[entity] + "\n")

        with open(self.processed_dir + '/id_entity_other', "w") as w:
            w.write("id      entity     class\n")
            for entity in node_list_other:
                w.write(str(node_dict[entity]) + "  " + entity + "  " + node_type[entity] + "\n")

        with open(self.processed_dir + '/id_class', "w") as w:
            w.write("id      class\n")
            for key in class_dict:
                w.write(str(class_dict[key]) + "  " + key + "\n")

        with open(self.processed_dir + '/id_relation', "w") as w:
            w.write("  id      relation\n")
            for relation in edge_dict.keys():
                w.write(str(edge_dict[relation]) + "  " + relation + "\n")

        edge_index = [[], []]
        edge_type = []
        train_idx = []
        train_y = []
        test_idx = []
        test_y = []

        for triple in graph:
            edge_index[0].append(node_dict[triple[0]])
            edge_index[1].append(node_dict[triple[2]])
            edge_type.append(edge_dict[triple[1]])

        add_edge_index_head = edge_index[1].copy()
        add_edge_index_tail = edge_index[0].copy()

        num_relations = max(edge_type) + 1
        add_edge_type = [i + num_relations for i in edge_type]
        edge_index[0] += add_edge_index_head
        edge_index[1] += add_edge_index_tail
        edge_type += add_edge_type

        for node in node_list_train:
            train_idx.append(node_dict[node])
            train_y.append(class_dict[node_type[node]])

        for node in node_list_test:
            test_idx.append(node_dict[node])
            test_y.append(class_dict[node_type[node]])

        data = Data(edge_index=torch.tensor(edge_index, dtype=torch.long))
        data.edge_type = torch.tensor(edge_type, dtype=torch.long)
        data.train_idx = torch.tensor(train_idx, dtype=torch.long)
        data.train_y = torch.tensor(train_y, dtype=torch.long)
        data.test_idx = torch.tensor(test_idx, dtype=torch.long)
        data.test_y = torch.tensor(test_y, dtype=torch.long)
        data.num_nodes = torch.tensor(edge_index, dtype=torch.long).max().item() + 1

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{"FB15K237_" + self.relation_name}()'
