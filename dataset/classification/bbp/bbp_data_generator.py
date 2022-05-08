from deepchem import molnet
lists, dataset, transformers = molnet.load_bbbp(featurizer = 'ECFP', splitter=None)  
Smiles, labels = dataset[0].ids, dataset[0].y

percent_training_data = 0.8
splitter = round(Smiles.shape[0] * percent_training_data)

train_Smiles = Smiles[:splitter]
test_Smiles = Smiles[splitter:]

train_labels = labels[:splitter]
test_labels = labels[splitter:]

with open ("data_test.txt", 'w') as output:
    i=0
    for row in test_Smiles:
        output.write(row + ' ' + str(test_labels[i])[1]  + '\n')
        i += 1
        
        
with open ("data_train.txt", 'w') as output:
    i=0
    for row in train_Smiles:
        output.write(row + ' ' + str(train_labels[i])[1]  + '\n')
        i += 1