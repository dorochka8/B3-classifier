import torch
import torch_geometric
import rdkit.Chem as Chem
from collections import Counter 

from config import config


class B3DBDataset(torch_geometric.data.Dataset):
    def __init__(self, dataframe):
        super(B3DBDataset, self).__init__()
        self.dataframe = dataframe
        self.processed_data = [self.process_smiles(row['SMILES'], row['BBB+/BBB-']) for _, row in dataframe.iterrows()]
        self.processed_data = [data for data in self.processed_data if data is not None]
        self.verify_data()

    def process_smiles(self, smiles, label):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atom_features = []
        edge_index = []

        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                atom.GetIsAromatic(),
                atom.GetImplicitValence(),
                atom.GetNumExplicitHs(),
            ]
            one_hot = self.get_atom_type_one_hot(atom.GetSymbol())
            features.extend(one_hot)
            atom_features.append(features)

        atom_features = torch.tensor(atom_features, dtype=torch.float)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        label = 1 if label == 'BBB+' else 0

        data = torch_geometric.data.Data(x=atom_features, edge_index=edge_index, y=torch.tensor(label, dtype=torch.float))

        return data

    def get_atom_type_one_hot(self, symbol):
        atom_symbols = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        one_hot = [0] * len(atom_symbols)
        if symbol in atom_symbols:
            one_hot[atom_symbols.index(symbol)] = 1
        return one_hot

    def len(self):
        return len(self.processed_data)

    def get(self, idx):
        return self.processed_data[idx]

    def verify_data(self):
        labels = [data.y.item() for data in self.processed_data]
        print("Label distribution:", Counter(labels))


def create_loader(dataset, val=True, test=True):
    train_size = int(0.8 * len(dataset))
    val_size   = int(0.1 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True )
    
    if val:  val_loader  = torch_geometric.loader.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
    if test: test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
    
    return train_loader, val_loader, test_loader