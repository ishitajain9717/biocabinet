from scripts.enrichment.gnn_data import *

data = GNNData(ppi_path='/Users/ishitajain/Desktop/9606.protein.actions.v11.0.txt', skip_head=True, exclude_protein_path=None)
data.get_protein_aac('/Users/ishitajain/Desktop/protein.STRING_all_connected.sequences.dictionary.tsv')


