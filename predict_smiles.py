"""Headless version of openpom/predict_odors.py.

predict_odors.py opens a Tk window; this script takes SMILES on the command
line and prints the top smells, so it works over SSH or in a notebook.

    python predict_smiles.py CCO
    python predict_smiles.py "CC(=O)OC1=CC=CC=C1C(=O)O" --top 15
"""

import argparse
from pathlib import Path

import deepchem as dc
import numpy as np

from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.models.mpnn_pom import MPNNPOMModel

TASKS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
    'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
    'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
    'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
    'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
    'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
    'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
    'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
    'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
    'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
    'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
    'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
    'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
    'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
    'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
    'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
    'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
    'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
    'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]

REPO = Path(__file__).resolve().parent / "openpom"
ENSEMBLE_DIR = REPO / "models" / "ensemble_models"
N_MODELS = 10


def load_models(device):
    models = []
    for i in range(N_MODELS):
        ckpt = ENSEMBLE_DIR / f"experiments_{i + 1}" / "checkpoint2.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        model = MPNNPOMModel(
            n_tasks=len(TASKS),
            batch_size=128,
            class_imbalance_ratio=None,
            loss_aggr_type='sum',
            node_out_feats=100,
            edge_hidden_feats=75,
            edge_out_feats=100,
            num_step_message_passing=5,
            mpnn_residual=True,
            message_aggregator_type='sum',
            mode='classification',
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            n_classes=1,
            readout_type='set2set',
            num_step_set2set=3,
            num_layer_set2set=2,
            ffn_hidden_list=[392, 392],
            ffn_embeddings=256,
            ffn_activation='relu',
            ffn_dropout_p=0.12,
            ffn_dropout_at_input_no_act=False,
            weight_decay=1e-5,
            self_loop=False,
            optimizer_name='adam',
            log_frequency=32,
            model_dir=str(ENSEMBLE_DIR / f"experiments_{i + 1}"),
            device_name=device,
        )
        model.restore(str(ckpt))
        models.append(model)
    return models


def predict(models, smiles):
    featurized = GraphFeaturizer().featurize([smiles])
    preds = [m.predict(dc.data.NumpyDataset(featurized)) for m in models]
    return np.mean(np.asarray(preds), axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles", nargs="+", help="one or more SMILES strings")
    parser.add_argument("--top", type=int, default=10, help="how many smells to print (default 10)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default cpu)")
    args = parser.parse_args()

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    models = load_models(device)
    for smiles in args.smiles:
        scores = np.asarray(predict(models, smiles)).squeeze()
        order = np.argsort(scores)[::-1][: args.top]
        print(f"\n{smiles}")
        print("-" * 40)
        for idx in order:
            print(f"{TASKS[idx]:<15} {scores[idx]:.4f}")


if __name__ == "__main__":
    main()
