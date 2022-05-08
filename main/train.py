#To start rdkit environment, please copy and paste before the simulation
#conda activate my-rdkit-env
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import preprocess as pp


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output,adjacencies):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        #print(self.embed_fingerprint)
        self.adjacencies = adjacencies
        #print(self.adjacencies)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)]) 
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        if task == 'classification':
            self.W_property = nn.Linear(dim, 2)


    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors

    def mlp(self, vectors):
        """Classifier or regressor based on multilayer perceptron."""
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs

    def forward_classifier(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_labels = torch.cat(data_batch[-1])

        if train:
            molecular_vectors = self.gnn(inputs)
            predicted_scores = self.mlp(molecular_vectors)
            loss = F.cross_entropy(predicted_scores, correct_labels)
            return loss
        else:
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_scores = self.mlp(molecular_vectors)
            predicted_scores = predicted_scores.to('cpu').data.numpy()
            predicted_scores = [s[1] for s in predicted_scores]
            correct_labels = correct_labels.to('cpu').data.numpy()
            return predicted_scores, correct_labels



class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = palm (model)
#        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            if task == 'classification':
                loss = self.model.forward_classifier(data_batch, train=True)
            palm(model)
            #self.optimizer.zero_grad()
            loss.backward()
            #self.optimizer.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_classifier(self, dataset):
        N = len(dataset)
        P, C = [], []
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            predicted_scores, correct_labels = self.model.forward_classifier(
                                               data_batch, train=False)
            P.append(predicted_scores)
            C.append(correct_labels)
        AUC = roc_auc_score(np.concatenate(C), np.concatenate(P))
        return AUC


    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

def proximal_l0(yvec, c):
    yvec_abs =  torch.abs(yvec)
    csqrt = torch.sqrt(c)
    
    xvec = (yvec_abs>=csqrt)*yvec
    return xvec

# solution for ||x-y||^2_2 + c||x||_2^2
def proximal_l2(yvec, c):
    return (1./(1.+c))*yvec
    

def palm(model, reg_l0=0.0001, reg_decay=0.0001, lr=0.001, lip=0.001):

    #adj_tensor = torch.tensor(model.adjacencies, requires_grad = True)
    
    #average_f = 0
    #average_o = 0
    
    adj_prune = model.adjacencies[0]
#    print(model.adjacencies)
    

    for name, param in model.named_parameters():
        if "W_fingerprint" in name:
            if(name[16:17] == 'w'): # being sure that we are not taking biases
                #average_f = torch.add(average_f,param)
                # there should be masking before proximal_l0
                #print('After Params :', param)
                if not (param.grad == None): 
                    param_tmp = param - lip * param.grad 
                    param = proximal_l0(param_tmp, torch.tensor(reg_l0))
        
        elif "W_out" in name: # output weights
            if(name[16:17] == 'w'):
                # other param weigth decay
                # average_o = torch.add(average_o,param)
                if not (param.grad == None): 
                    param_tmp = param - lr*param.grad
                    param = proximal_l2(param_tmp, torch.tensor(reg_decay))
           
                

    



if __name__ == "__main__":

    #(task, dataset, radius, dim, layer_hidden, layer_output, batch_train, batch_test, lr, lr_decay, decay_interval, iteration, setting) = sys.argv[1:]
    task = 'classification'
    dataset = 'hiv' # or 'bbp'
    radius = 1
    dim = 50
    layer_hidden = 6
    layer_output = 6
    batch_train = 32
    batch_test = 32
    lr = 1e-4
    lr_decay = 0.99
    decay_interval = 10
    iteration = 1000
     
    setting = task[0:5] + '_' + dataset + '_' + str(radius) + '_' + str(dim) + '_' + str(layer_hidden) + '_' + str(layer_output) + '_' + str(batch_train)  + '_' + str(batch_test) + '_' + str(lr) + '_' + str(lr_decay) + '_' + str(decay_interval) + '_' + str(iteration)
     
    (radius, dim, layer_hidden, layer_output, batch_train, batch_test, decay_interval, iteration) = map(int, [radius, dim, layer_hidden, layer_output,
                            batch_train, batch_test,
                            decay_interval, iteration])
    lr, lr_decay = map(float, [lr, lr_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-'*100)

    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    (dataset_train, dataset_dev, dataset_test,
     N_fingerprints, adjacencies) = pp.create_datasets(task, dataset, radius, device)
    print('-'*100)

    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)

    print('Creating a model.')
    torch.manual_seed(1234)
    model = MolecularGraphNeuralNetwork(
            N_fingerprints, dim, layer_hidden, layer_output,adjacencies).to(device)
    
    torch.set_printoptions(threshold=10_000)

    print(len(model.adjacencies))
    
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    file_result = '../output/result--' + setting + '.txt'
    if task == 'classification':
        result = 'Epoch\tTime(sec)\tLoss_train\tAUC_dev\tAUC_test'

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)

    start = timeit.default_timer()

    for epoch in range(iteration):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)

        if task == 'classification':
            prediction_dev = tester.test_classifier(dataset_dev)
            prediction_test = tester.test_classifier(dataset_test)


        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_train,
                                     prediction_dev, prediction_test]))
        tester.save_result(result, file_result)

        print(result)
