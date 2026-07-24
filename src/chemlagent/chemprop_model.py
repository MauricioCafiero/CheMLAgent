"""Chemprop data + MPNN wrapper for the AZO Lmax task.

Adapted from the starter ``../arch/CafChemProp.py`` (MauricioCafiero/CafChem), with
one rigor change: instead of chemprop's own *random* ``data.make_split_indices``
split, the data class ingests the **pre-exported seed-132 Murcko scaffold split**
(see export_split.py / data/scaffold_split_seed132.json) so Chemprop is
trained/tested on the *same* 534/87 molecules the Mordred SVR/MLP used, making
the external new-azo MAE directly comparable.

Two classes:
  - ``chemprop_data``: builds MoleculeDatapoints / datasets from pre-split
    SMILES+targets, normalizes targets (train scaler reused on val), and builds
    dataloaders (``featurize`` once, then ``make_loaders(batch_size)`` per
    config so a sweep can vary batch size without re-featurizing).
  - ``chemprop_model``: constructs the MPNN with ALL sweepable knobs exposed
    (message-passing type/depth/width/dropout/undirected, aggregation, FFN
    width/layers/dropout, the Noam-like LR schedule's init/max/final + warmup),
    trains with a lightning Trainer + ModelCheckpoint on val_loss, runs
    inference, and saves/loads the model.
"""
import contextlib
import logging
import os
import warnings

import numpy as np
import torch
import urllib.request
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from chemprop import data, featurizers, models, nn
from chemprop.models.model import MPNN

# Broadcast print flag: the agent sets chemlagent.chemprop_model.print_flag =
# <args.print> at startup (see agent.py, mirroring src/agent_template.py). The
# chemprop tool banner lives in tools.py and always prints; everything here
# (init notices, download/load/featurize progress, save messages) is debug
# output and is gated. Default False = debug-silent unless --print.
print_flag = False


# Third-party loggers (lightning's rank_zero_info / MPS-available notices,
# chemprop, torch UserWarnings, RDKit) are NOT our print() statements, so the
# print_flag gate above does not touch them -- they flood stderr during
# training/inference. _stealth_mode muffles them when --print is off (banners in
# tools.py still print by convention) and restores prior levels on exit. When
# --print is on it is a no-op so a verbose run still shows everything.
_STEALTH_LOGGERS = ("lightning", "lightning_fabric", "lightning.pytorch",
                   "pytorch_lightning", "chemprop", "torch", "rdkit")


@contextlib.contextmanager
def _stealth_mode():
    if print_flag:
        yield
        return
    prev = {n: logging.getLogger(n).level for n in _STEALTH_LOGGERS}
    for n in _STEALTH_LOGGERS:
        logging.getLogger(n).setLevel(logging.ERROR)
    rdkit_off = False
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        rdkit_off = True
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            for n, lvl in prev.items():
                logging.getLogger(n).setLevel(lvl)
            if rdkit_off:
                try:
                    from rdkit import RDLogger
                    RDLogger.EnableLog("rdApp.*")
                except Exception:
                    pass


class _MPSCacheFlush(Callback):
    """Hand MPS's freed buffer pool back to the OS after each epoch.

    On Apple Silicon, torch's MPS backend retains freed allocations in an
    internal cache rather than returning them to the OS, so resident memory
    climbs across epochs and can pressure the system (the laptop froze during a
    chemprop run before this). empty_cache() releases that pool. It is a no-op
    on CPU (torch.mps is absent), so this callback is always safe to attach.
    """

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _mps_empty_cache():
    """Best-effort MPS cache flush; no-op on CPU or if MPS isn't built."""
    try:
        torch.mps.empty_cache()
    except Exception:
        pass


# ---- CheMeleon foundation-model weights ----
# CheMeleon (arXiv:2506.15792) pre-trains a chemprop BondMessagePassing
# (d_h=2048, depth=6, ~8.7M params) on ~1M PubChem SMILES via low-noise
# classical descriptors. Fine-tuning reuses that MP and attaches a fresh
# regression FFN head. chemprop >=2.2.0 supports it via ``--from-foundation
# CHEMELEON`` (CLI auto-download); since we drive the Python API, we fetch the
# weights ourselves once. URL mirrors the one in chemprop/cli/train.py.
_CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def _load_foundation_mp(foundation_path):
    """Load (downloading once if missing) a foundation message-passing block.

    ``foundation_path`` is a local .pt file holding ``{'hyper_parameters',
    'state_dict'}`` (the CheMeleon format). Returns a ready ``BondMessagePassing``
    with the foundation weights loaded. Downstream featurization is unchanged
    (standard ``SimpleMoleculeMolGraphFeaturizer``); only the MP initialization
    differs from a from-scratch run.
    """
    if not os.path.exists(foundation_path):
        os.makedirs(os.path.dirname(os.path.abspath(foundation_path)) or '.',
                    exist_ok=True)
        if print_flag:
            print(f'Downloading CheMeleon foundation weights from Zenodo to '
                  f'{foundation_path} ...')
        urllib.request.urlretrieve(_CHEMELEON_URL, foundation_path)
        if print_flag:
            print('  done.')
    ck = torch.load(foundation_path, weights_only=True)
    mp = nn.BondMessagePassing(**ck['hyper_parameters'])
    mp.load_state_dict(ck['state_dict'])
    if print_flag:
        print(f'Loaded foundation MP from {foundation_path} '
              f'(d_h={ck["hyper_parameters"].get("d_h")}, '
              f'depth={ck["hyper_parameters"].get("depth")}, '
              f'output_dim={mp.output_dim})')
    return mp


class chemprop_data:
    """Read pre-split data, featurize to mol-graphs, build dataloaders."""

    def __init__(self, num_workers=0):
        self.num_workers = num_workers
        if print_flag:
            print('Class chemprop_data initialized')

    @staticmethod
    def _datapoints(smis, ys):
        # Keep targets 2-D (n, n_tasks) and pass each ROW (a 1-element array for
        # single-task regression) to from_smi, so the dataset's _Y is 2-D and
        # StandardScaler.fit (sklearn >=1.9 rejects 1-D) is happy. Matching the
        # starter CafChemProp, which passed a (n,1) target matrix.
        if ys is None:
            rows = [None] * len(smis)
        else:
            arr = np.asarray(ys, dtype=float).reshape(-1, 1)
            rows = list(arr)
        return [data.MoleculeDatapoint.from_smi(s, y) for s, y in zip(smis, rows)]

    def load_pre_split(self, train_smiles, train_y, val_smiles, val_y,
                       test_smiles, test_y):
        """Ingest the exported scaffold split (train/val/test) directly.

        val is a fixed-seed carve out of train (see run_chemprop.py); the test
        set here is the 87-molecule scaffold holdout, identical to the Mordred
        models. Targets are raw Lmax (no log).
        """
        self.train_data = [self._datapoints(train_smiles, train_y)]
        self.val_data = [self._datapoints(val_smiles, val_y)]
        self.test_data = [self._datapoints(test_smiles, test_y)]
        if print_flag:
            print(f'Data loaded: train={len(self.train_data[0])} '
                  f'val={len(self.val_data[0])} test={len(self.test_data[0])}')

    def featurize(self):
        """Featurize to mol-graphs and normalize targets (train scaler reused
        on val). Call ONCE; then build per-batch-size loaders with
        make_loaders(). Returns the train StandardScaler (also wired into the
        model's UnscaleTransform so predictions come back in raw nm).
        """
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.featurizer = featurizer

        self.train_dset = data.MoleculeDataset(self.train_data[0], featurizer)
        scaler = self.train_dset.normalize_targets()

        self.val_dset = data.MoleculeDataset(self.val_data[0], featurizer)
        self.val_dset.normalize_targets(scaler)

        self.test_dset = data.MoleculeDataset(self.test_data[0], featurizer)
        if print_flag:
            print('Data featurized.')
        return scaler

    def make_loaders(self, batch_size=64):
        """Build train (shuffled) / val / test dataloaders, plus a non-shuffled
        twin of the train loader for evaluation (the training loader shuffles,
        so predicting on it mis-orders predictions vs dataset-order targets)."""
        self.train_loader = data.build_dataloader(self.train_dset, batch_size=batch_size,
                                                 num_workers=self.num_workers)
        self.train_eval_loader = data.build_dataloader(self.train_dset, batch_size=batch_size,
                                                       num_workers=self.num_workers, shuffle=False)
        val_loader = data.build_dataloader(self.val_dset, batch_size=batch_size,
                                           num_workers=self.num_workers, shuffle=False)
        test_loader = data.build_dataloader(self.test_dset, batch_size=batch_size,
                                            num_workers=self.num_workers, shuffle=False)
        return self.train_loader, val_loader, test_loader

    def get_full_dsets(self):
        """Return the RAW (un-scaled) target values of train/val/test, for R²/
        MAE on the original nm scale. NB: must read ``_Y`` (the cached raw
        targets), NOT ``self[i].y`` -- the latter comes from ``Datum`` built
        with the *scaled* ``self.Y`` after normalize_targets (train/val), while
        predictions return in raw nm via UnscaleTransform, so the two would
        mismatch. ``_Y`` reads each datapoint's original ``d.y`` and is never
        mutated by normalization."""
        full_train = list(self.train_dset._Y.reshape(-1))
        full_val = list(self.val_dset._Y.reshape(-1))
        full_test = list(self.test_dset._Y.reshape(-1))
        return full_train, full_val, full_test

    def make_new_dataloader(self, new_smiles, batch_size=64):
        """Build a (no-target) dataloader for the external new-azo holdout."""
        new_data = [data.MoleculeDatapoint.from_smi(s) for s in new_smiles]
        new_dset = data.MoleculeDataset(new_data, featurizer=self.featurizer)
        return data.build_dataloader(new_dset, batch_size=batch_size, shuffle=False)


class chemprop_model:
    """Construct, train, and run inference with a Chemprop MPNN.

    All sweepable knobs are constructor args (defaults reproduce the original
    chemprop defaults / the first single run): message-passing type/depth/width/
    dropout/undirected, aggregation, FFN width/layers/dropout, and the Noam-like
    LR schedule (init_lr -> max_lr -> final_lr over warmup..cooldown).
    """

    def __init__(self, mess_pass='bond', aggre='mean', batch_norm=True,
                 depth=3, d_h=300, mp_dropout=0.0, undirected=False,
                 activation='relu', ffn_hidden_dim=300, ffn_n_layers=1,
                 ffn_dropout=0.0, max_lr=1e-3, init_lr=1e-4, final_lr=1e-4,
                 warmup_epochs=2, from_foundation=None, foundation_path=None,
                 freeze_mp=False):
        self.mess_pass = mess_pass
        self.aggre = aggre
        self.batch_norm = batch_norm
        self.depth = depth
        self.d_h = d_h
        self.mp_dropout = mp_dropout
        self.undirected = undirected
        self.activation = activation
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_n_layers = ffn_n_layers
        self.ffn_dropout = ffn_dropout
        self.max_lr = max_lr
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.from_foundation = from_foundation
        self.freeze_mp = freeze_mp

        if from_foundation is not None:
            # Foundation-MP path: the message-passing block is initialized from
            # a pretrained CheMeleon checkpoint. depth/d_h/mp_dropout/
            # undirected/activation/mess_pass are baked into the checkpoint's
            # hyper_parameters and the from-scratch kwargs are ignored (mirrors
            # chemprop CLI's ``--from-foundation`` behavior). mess_pass must be
            # bond -- CheMeleon is a BondMessagePassing.
            if mess_pass != 'bond':
                raise ValueError("from_foundation requires mess_pass='bond' "
                                 "(CheMeleon is a BondMessagePassing).")
            if foundation_path is None:
                raise ValueError("from_foundation requires foundation_path.")
            self.mp = _load_foundation_mp(foundation_path)
            if print_flag:
                print('  (ignored by foundation path: depth, d_h, mp_dropout, '
                      'undirected, activation -- taken from checkpoint)')
            if freeze_mp:
                # Freeze the pretrained MP: train only the FFN head. Docs note
                # this can sometimes help. Default is continue-training.
                self.mp.eval()
                self.mp.apply(lambda m: m.requires_grad_(False))
                if print_flag:
                    print('  Foundation MP FROZEN (training FFN head only).')
            else:
                if print_flag:
                    print('  Foundation MP will be fine-tuned (continue training).')
        else:
            mp_cls = nn.BondMessagePassing if mess_pass == 'bond' else nn.AtomMessagePassing
            if mess_pass not in ('bond', 'atom'):
                raise ValueError(f'unknown message passing type: {mess_pass}')
            self.mp = mp_cls(d_h=self.d_h, depth=self.depth, dropout=self.mp_dropout,
                             undirected=self.undirected, activation=self.activation)

        agg_map = {'mean': nn.MeanAggregation, 'sum': nn.SumAggregation,
                   'norm': nn.NormAggregation}
        if self.aggre not in agg_map:
            raise ValueError(f'unknown aggregation: {self.aggre}')
        self.agg = agg_map[self.aggre]()
        if print_flag:
            print('Class chemprop_model initialized')

    def construct_model(self, scaler, metrics=('mae', 'r2')):
        """Assemble the MPNN. The FFN input dim must match the aggregation
        output: mp.output_dim for mean/sum, +1 for norm (NormAggregation
        appends a graph-norm feature). Using ``self.mp.output_dim`` (not
        ``self.d_h``) keeps this correct for both the from-scratch path (where
        output_dim == d_h) and the foundation path (CheMeleon output_dim=2048,
        independent of the ignored ``d_h`` kwarg). The UnscaleTransform undoes
        train target standardization at prediction -> outputs in raw nm."""
        metrics_hash = {'mse': nn.metrics.MSE, 'rmse': nn.metrics.RMSE,
                        'mae': nn.metrics.MAE, 'r2': nn.metrics.R2Score}
        metric_list = [metrics_hash[m]() for m in metrics if m in metrics_hash]
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        agg_out_dim = self.mp.output_dim + (1 if self.aggre == 'norm' else 0)
        self.ffn = nn.RegressionFFN(input_dim=agg_out_dim,
                                   hidden_dim=self.ffn_hidden_dim,
                                   n_layers=self.ffn_n_layers,
                                   dropout=self.ffn_dropout,
                                   output_transform=output_transform)
        self.mpnn = models.MPNN(self.mp, self.agg, self.ffn, self.batch_norm,
                                metric_list, warmup_epochs=self.warmup_epochs,
                                init_lr=self.init_lr, max_lr=self.max_lr,
                                final_lr=self.final_lr)
        return self.mpnn

    def train_model(self, train_loader, val_loader, epochs=50, devices=1,
                    checkpoint_dir='checkpoints', accelerator='auto'):
        """Fit with a lightning Trainer; ModelCheckpoint keeps the best-val-loss
        epoch (restored before test/inference). Returns the best val_loss
        (the sweep's selection metric).

        ``accelerator`` defaults to 'auto' (MPS on Apple Silicon when
        available). Pass 'cpu' to force CPU -- the cp312 torch wheel segfaults
        inside chemprop/lightning on MPS, so the CheMLAgent tool layer forces
        'cpu' by default (see tools.train_chemprop).

        All third-party logging/warnings during fit are gated behind
        print_flag via _stealth_mode (the tool banner in tools.py still prints);
        the _MPSCacheFlush callback releases MPS's freed-buffer pool back to
        the OS each epoch so resident memory does not climb across epochs.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpointing = ModelCheckpoint(checkpoint_dir, 'best-{epoch}-{val_loss:.2f}',
                                        monitor='val_loss', mode='min', save_last=True)
        with _stealth_mode():
            self.trainer = pl.Trainer(
                logger=False, enable_checkpointing=True, enable_progress_bar=False,
                accelerator=accelerator, devices=devices, max_epochs=epochs,
                callbacks=[checkpointing, _MPSCacheFlush()])
            self.trainer.fit(self.mpnn, train_loader, val_loader)
            best_val_loss = float('nan')
            if checkpointing.best_model_score is not None:
                best_val_loss = float(checkpointing.best_model_score.item())
            best = checkpointing.best_model_path
            if best and os.path.exists(best):
                self.mpnn = MPNN.load_from_checkpoint(best)
                if print_flag:
                    print(f'  Restored best checkpoint: {os.path.basename(best)} '
                          f'(val_loss={best_val_loss:.4f})')
            _mps_empty_cache()
        return best_val_loss

    def get_preds(self, loader, accelerator='auto'):
        """Run inference on a dataloader; return a flat list of predictions
        (raw nm, thanks to the UnscaleTransform). ``accelerator`` mirrors
        train_model; the tool layer forces 'cpu' by default. Third-party
        logging/warnings are gated behind print_flag via _stealth_mode."""
        with _stealth_mode():
            with torch.inference_mode():
                inftrainer = pl.Trainer(logger=None, enable_progress_bar=False,
                                        accelerator=accelerator, devices=1)
            raw_preds = inftrainer.predict(self.mpnn, loader)
            _mps_empty_cache()
        return [float(v[0].item()) for sub in raw_preds for v in sub]

    def save_model(self, path):
        torch.save({'hyper_parameters': self.mpnn.hparams,
                    'state_dict': self.mpnn.state_dict()}, path)
        if print_flag:
            print(f'Model saved to {path}')

    def load_model(self, path):
        self.mpnn = MPNN.load_from_file(path)
        return self.mpnn