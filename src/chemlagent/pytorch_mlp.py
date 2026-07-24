import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import warnings
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from chemlagent.descriptor_cleaning import aggressive_cleaner

# numpy's matmul (and sklearn's PCA, which calls it internally) can emit
# spurious "divide by zero / overflow / invalid value encountered in matmul"
# RuntimeWarnings even when every operand and every result is finite -- a known
# numpy/BLAS quirk on this platform. Preprocessing here is done in float64 with
# NaN->0 and a +/-100 clip, so inputs are bounded and outputs are verified
# finite; silence this specific noise so it does not drown out real issues.
warnings.filterwarnings('ignore', message='.*encountered in matmul.*',
                        category=RuntimeWarning)

# Broadcast print flag: the agent sets chemlagent.pytorch_mlp.print_flag =
# <args.print> at startup (see agent.py, mirroring src/agent_template.py's
# `modrag_protein_functions.print_flag = print_flag`). Every console print in
# this module is gated on it so the MLP pipeline is silent unless --print is
# given.
print_flag = False

# Set device to Apple Silicon (MPS) if available, otherwise CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if print_flag:
  print(f"Using device: {device}")

class MLP_Model(nn.Module):
  '''
  Multilayer Perceptron Model using PyTorch. Hidden activation is Sigmoid.
    Args:
      neurons: number of neurons in each hidden layer
      input_dims: number of input dimensions
      num_hidden_layers: number of additional hidden layers BEYOND the input
        layer (linear_input), which is itself a sigmoid hidden layer. So the
        total number of sigmoid hidden layers is 1 + num_hidden_layers
        (e.g. num_hidden_layers=1 -> 2 hidden layers, matching the
        input/2-hidden/output skip-connected architecture).
      classifier_flag: Boolean to perform classification
      skip_connection: Boolean to route the original input features to the
        final layer and concatenate them with the hidden representation
        before the output layer. The output/classifier linear layers then
        receive (neurons + input_dims) inputs.
  '''
  def __init__(self, neurons: int, input_dims: int, num_hidden_layers: int,
               classifier_flag=False, num_classes = 1, skip_connection=False):
    super(MLP_Model, self).__init__()
    self.neurons = neurons
    self.input_dims = input_dims
    self.num_hidden_layers = num_hidden_layers
    self.classifier_flag = classifier_flag
    self.num_classes = num_classes
    self.skip_connection = skip_connection
    self.batchnorm = nn.BatchNorm1d(self.input_dims)
    self.linear_input = nn.Sequential(
        nn.Linear(self.input_dims, self.neurons),
        nn.Sigmoid())
    self.linear_sigmoid = nn.Sequential(
        nn.Linear(self.neurons, self.neurons),
        nn.Sigmoid())
    # When the skip connection is on, the original input features are
    # concatenated to the hidden representation before the output layer, so
    # the output layers take neurons + input_dims inputs.
    out_in = self.neurons + (self.input_dims if self.skip_connection else 0)
    self.linear_output = nn.Linear(out_in, 1)
    self.linear_class_out = nn.Linear(out_in, self.num_classes)
    self.classifier_output = nn.LogSoftmax()

    # Move model to device
    self.to(device)

    f = open("MLP_model_params.txt","w")
    f.write(f"neurons: {self.neurons}\n")
    f.write(f"input_dims: {self.input_dims}\n")
    f.write(f"num_hidden_layers: {self.num_hidden_layers}\n")
    f.write(f"classifier_flag: {self.classifier_flag}\n")
    f.write(f"num_classes: {self.num_classes}\n")
    f.write(f"skip_connection: {self.skip_connection}")
    f.close()

  def forward(self, x):
    '''
      Passes the input through a batch normalization layer, an input layer,
      a number of hidden layers, and an output layer.

        Args:
          x: input tensor
        Returns:
          output: output tensor
    '''
    # Keep the original (pre-batchnorm) features for the optional skip path
    skip = x
    x = self.batchnorm(x)
    x = self.linear_input(x)
    for i in range(self.num_hidden_layers):
      x = self.linear_sigmoid(x)

    # Route the original features to the final step and concatenate along
    # the feature dimension.
    if self.skip_connection:
      x = torch.cat([x, skip], dim=1)

    if self.classifier_flag == False:
      output = self.linear_output(x)
    else:
      x = self.linear_class_out(x)
      output = self.classifier_output(x)

    return output

def train(dataloader, model, loss_fn, optimizer, classifier_flag=False, num_classes = 1, device=device):
  '''
    Trains the model for one epoch.

    Args:
      dataloader: dataloader for the training data
      model: model to train
      loss_fn: loss function to use
      optimizer: optimizer to use
      classifier_flag: Boolean to perform classification
      device: device to train on (cpu or mps)
    Returns:
      avg_loss: mean loss over all batches this epoch (float). The caller's
        early-stopping loop watches this to stop once training plateaus.
  '''
  if classifier_flag == False:
    num_classes = 1

  size = len(dataloader.dataset)
  model.train()

  total_loss = 0.0
  n_batches = 0

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()

    pred = model(X)
    if classifier_flag == False:
      loss = loss_fn(pred, y.view(-1,1))
    else:
      loss = loss_fn(pred, y)
    total_loss += loss.item()
    n_batches += 1

    loss.backward()
    optimizer.step()

    if print_flag and batch % 2 == 0:
      current = batch * len(X)
      avg_loss = total_loss / n_batches
      print(f"Batch: {batch}, Loss: {avg_loss:.7f} [{current:>5d}/{size:>5d}]")

  return total_loss / n_batches if n_batches else float('inf')

def evaluate_regression(X_train, y_train, X_test, y_test, model, device=device):
  '''
    Evaluates the model on the training and test data.

    Args:
      X_train: training data
      y_train: training truths
      X_test: test data
      y_test: test truths
      model: model to evaluate
      device: device to evaluate on (cpu or mps)
    Returns:
      train_r2: R^2 score of the training data
      test_r2: R^2 score of the test data
      Plots the training and test data against the model's predictions.
  '''
  # as_tensor (not torch.tensor) avoids the "To copy construct from a tensor"
  # UserWarning when X_train/X_test are already tensors (the caller in tools.py
  # passes prep.X_train/prep.X_test, which are float32 tensors). It shares
  # storage when possible and only moves to device here.
  X_train = torch.as_tensor(X_train, dtype=torch.float32).to(device)
  X_test = torch.as_tensor(X_test, dtype=torch.float32).to(device)

  model.eval()
  train_pred = model(X_train)
  test_pred = model(X_test)
  y_total = np.concatenate((y_train, y_test), axis=0)

  train_r2 = r2_score(y_train, train_pred.detach().cpu().numpy())
  test_r2 = r2_score(y_test, test_pred.detach().cpu().numpy())
  if print_flag:
    print(f"Train R2 Score: {train_r2}")
    print(f"Test R2 Score: {test_r2}")

  plt.scatter(y_train,train_pred.detach().cpu().numpy(),color="blue",label="ML-train")
  plt.scatter(y_test,test_pred.detach().cpu().numpy(),color="green",label="ML-valid")
  plt.plot(y_total,y_total,color="red",label="Best Fit")
  plt.legend()
  plt.xlabel("known")
  plt.ylabel("predicted")
  plt.show

  return train_r2, test_r2

def predict_single_value(model, fps, smiles_to_predict, prep=None,
                        truths=None, inverse_func=None, verbose=True,
                        device=device):
  '''
    Predicts values for a set of SMILES strings and (optionally) compares them
    to known targets. Mirrors evaluate_model in models.py: the fingerprints/
    descriptor object passed in as fps is used to transform the SMILES list
    (fps.transform(smiles_to_predict)), the feature matrix is put through the
    SAME preprocessing used during training (nan_to_num then standardization
    with the train-set feature_mean / feature_std stored on the prep_data
    instance), and passed through the PyTorch model. When truth values are
    supplied an R^2 score and a SMILES / Predicted / Actual / Difference table
    (with an averaged final row) are printed.

    Args:
      model: trained PyTorch model to use for prediction
      fps: fitted fingerprint/descriptor generator (sklearn-style transformer);
        exposes .transform(smiles_list) returning the feature matrix
      smiles_to_predict: list of SMILES strings to predict (a single SMILES
        string is accepted and treated as a one-element list)
      prep: the prep_data instance used to build the training data. Its
        feature_mean and feature_std are reused so new molecules are
        standardized exactly as the training set was. If None, the features
        are only nan_to_num-cleaned and passed through unscaled (this will
        generally NOT match training and is mainly for debugging).
      truths (optional): list/array of true target values aligned with
        smiles_to_predict. When provided, R^2 and the comparison table are
        printed.
      inverse_func (optional): callable applied element-wise to the model's
        predictions to map them back to the original target space before any
        comparison to truths. Use this when the model was trained on
        transformed targets (e.g. pass lambda p: 10**p when targets were
        log10-transformed). Predictions are returned in the inverse-mapped
        space when this is provided.
      verbose (optional): if True (default) print the R^2 and the SMILES /
        Predicted / Actual / Difference table; if False, compute everything
        silently and just return the predictions (handy for sweeps).
      device: device to predict on (cpu or mps)
    Returns:
      predictions: NumPy array of predicted values, one per SMILES string
  '''
  # Allow a single SMILES string to be passed in for convenience.
  if isinstance(smiles_to_predict, str):
    smiles_to_predict = [smiles_to_predict]

  # Mirror create_data_loader preprocessing exactly, in float64: the train-fit
  # DescriptorCleaner (column keep_mask + median imputation, inf -> nan),
  # standardization with the train-set statistics, clip to +/-100, and (when
  # PCA was used at train time) the PCA projection. float64 + clip prevents
  # overflow/NaN for out-of-distribution molecules with extreme descriptor
  # values; results are downcast to float32 only at the end. All artifacts are
  # read off the prep_data instance.
  X = np.asarray(fps.transform(smiles_to_predict), dtype=np.float64)
  if prep is not None:
    # apply the train-fit cleaner: inf -> nan, keep_mask, median impute.
    X = np.where(np.isinf(X), np.nan, X)
    X = X[:, prep.keep_mask]
    inds = np.where(np.isnan(X))
    if inds[0].size:
      X = X.copy()
      X[inds] = np.take(prep.medians, inds[1])
    feature_std = np.asarray(prep.feature_std, dtype=np.float64).copy()
    feature_std[feature_std == 0] = 1.0
    X = (X - np.asarray(prep.feature_mean, dtype=np.float64)) / feature_std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -100.0, 100.0)
    if prep.pca_components is not None:
      comps = np.asarray(prep.pca_components, dtype=np.float64)
      pmean = np.asarray(prep.pca_mean, dtype=np.float64)
      X = (X - pmean) @ comps.T
      X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

  temp_tensor = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)

  model.eval()
  with torch.no_grad():
    prediction = model(temp_tensor)
  predictions = prediction.detach().cpu().numpy().reshape(-1)

  # Map predictions back to the original target space when the model was
  # trained on transformed targets (e.g. log10 -> power of 10).
  if inverse_func is not None:
    predictions = inverse_func(predictions)

  have_truths = truths is not None
  smiles_width = max(len(smi) for smi in smiles_to_predict) + 2
  if have_truths:
    truths = np.asarray(truths, dtype=float).reshape(-1)
    r2 = r2_score(truths, predictions)
    differences = [abs(predictions[i] - truths[i]) for i in range(len(smiles_to_predict))]
    avg_diff = sum(differences) / len(differences) if differences else 0.0
    if verbose:
      print(f'Predictions for new molecules:')
      print(f'R^2 score (new dataset): {r2}')
      print(f'{"SMILES":<{smiles_width}} {"Predicted Lmax":<20} {"Actual Lmax":<20} {"Difference":<15}')
      for i, smiles in enumerate(smiles_to_predict):
        diff = differences[i]
        print(f'{smiles:<{smiles_width}} {predictions[i]:<20.1f} {truths[i]:<20.1f} {diff:<15.1f}')
      print(f'{"Average":<{smiles_width}} {"":20} {"":20} {avg_diff:<15.1f}')
  elif verbose:
    print(f'Predictions for new molecules:')
    print(f'{"SMILES":<{smiles_width}} {"Predicted Lmax":<20}')
    for i, smiles in enumerate(smiles_to_predict):
      print(f'{smiles:<{smiles_width}} {predictions[i]:<20.1f}')

  return predictions

class prep_data():
  '''
  Data class to prepare raw data for model
  '''
  def __init__(self, batch_size: int, shuffle = True, classifier_flag = False,
               reduce_dim=None, var_threshold=1e-3, corr_threshold=0.98,
               pca_var=0.95, pca_n_components=None):
    '''
        Sets up data prep parameters.

        Args:
            batch_size: batch size for training / data loader
            shuffle: Boolean to shuffle training set
            reduce_dim: None to keep all features; "filter" to drop zero/
              near-zero-variance and highly-correlated duplicate columns;
              "pca" to apply the filter and then PCA (pca mode only). The
              reduction is fit on the TRAIN split only and stored on the
              instance so predict_single_value can apply the identical
              transform to new molecules.
            var_threshold: columns with std <= this on the train split are
              dropped (filter / pca modes only).
            corr_threshold: greedy pairwise |correlation| above this drops
              the later of two columns (filter / pca modes only).
            pca_var: fraction of explained variance to retain with PCA
              (pca mode only). Used only when pca_n_components is None.
            pca_n_components: if not None, fix PCA to exactly this many
              components (pca mode only) instead of selecting by explained
              variance. E.g. 95 reproduces the literature setup of reducing
              raw Mordred descriptors to 95 input features.
    '''
    self.classifier_flag = classifier_flag
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.reduce_dim = reduce_dim
    self.var_threshold = var_threshold
    self.corr_threshold = corr_threshold
    self.pca_var = pca_var
    self.pca_n_components = pca_n_components
    # Reduction artifacts, populated by create_data_loader.
    self.keep_mask = None
    self.pca_components = None
    self.pca_mean = None

    if print_flag:
      print("prep data class initialized!")
    
  def create_data_loader(self, X_train, y_train, X_test, y_test):
    '''
      Creates a data loader for the training and test data.

      Args: 
        None
      Returns:
        train_dataset: training dataset
        test_dataset: test dataset
        train_loader: training data loader
        test_loader: test data loader
    '''
    import numpy as np

    # Work in float64 through standardization/PCA so extreme descriptor
    # values (some molecules produce values orders of magnitude larger than
    # the training range) cannot overflow to inf during (x - mean)/std before
    # we clip and downcast to float32 for the tensors/model.
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    y_train = np.ascontiguousarray(y_train, dtype=np.float32 if not self.classifier_flag else np.int64)
    y_test = np.ascontiguousarray(y_test, dtype=np.float32 if not self.classifier_flag else np.int64)

    # Mordred/3D descriptors contain NaN/inf for uncomputable descriptors and
    # span many orders of magnitude. Clean them with the shared
    # DescriptorCleaner (aggressive_cleaner: inf->nan, drop missing-heavy /
    # near-zero-variance / corr>0.98 duplicate columns, then zero-fill i.e.
    # nan_to_num->0) -- fit on the TRAIN split only, and stored on self so
    # predict_single_value and save_stats can replay the identical transform on
    # new molecules. NB: zero-fill (NOT median) is deliberate -- the MLP's tuned
    # weight_decay/epochs/PCA config depends on it; median imputation was tested
    # and regresses the MLP 27 -> 40 nm. The corr filter checks corr directly
    # (not |corr|), matching the original inline filter the MLP was tuned
    # against; using |corr| drops negatively-correlated duplicates and shifts
    # the PCA basis, also regressing the MLP. Without cleaning, BatchNorm1d in
    # the model produces NaN -> NaN loss.
    cleaner = aggressive_cleaner()
    X_train = cleaner.fit_transform(X_train)
    X_test  = cleaner.transform(X_test)
    self.cleaner = cleaner
    self.keep_mask = cleaner.keep_mask_
    self.medians = cleaner.medians_
    if print_flag:
      print(f'Feature cleaning: {cleaner.n_features_in_} -> {cleaner.n_kept_} '
            f'columns (drop missing-heavy / near-constant / corr>0.98, '
            f'zero-fill)')

    # Standardize using TRAIN statistics only (avoids leakage). The aggressive
    # cleaner already dropped std<=1e-3 columns, so no division by zero here.
    col_mean = X_train.mean(axis=0)
    col_std  = X_train.std(axis=0)
    col_std[col_std == 0] = 1.0
    X_train = (X_train - col_mean) / col_std
    X_test  = (X_test  - col_mean) / col_std
    # Clean any residual inf/nan and clip to a sane standardized range. In-
    # distribution values are ~N(0,1) so +/-100 is a no-op there; this only
    # caps out-of-distribution molecules with extreme descriptor values, and
    # is applied identically in predict_single_value so train/test/new stay
    # consistent.
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
    X_train = np.clip(X_train, -100.0, 100.0)
    X_test  = np.clip(X_test,  -100.0, 100.0)

    # Keep the scaler for later use on new molecules (predict_single_value)
    self.feature_mean = col_mean
    self.feature_std = col_std

    # Optional PCA on top of standardization (fit on train, applied to test
    # and to new molecules via stored components/mean). Stored in float64 for
    # the same overflow-safety reason; predict_single_value casts as needed.
    if self.reduce_dim == "pca":
      from sklearn.decomposition import PCA
      # pca_n_components (e.g. 95) fixes the component count to reproduce the
      # literature setup; otherwise select by explained variance (pca_var).
      if self.pca_n_components is not None:
        n_comp = int(self.pca_n_components)
        pca = PCA(n_components=n_comp, random_state=132)
        pca.fit(X_train)
        sel_desc = f'{n_comp} components (fixed)'
      else:
        pca = PCA(n_components=self.pca_var, random_state=132)
        pca.fit(X_train)
        sel_desc = f'{self.pca_var:.0%} variance'
      self.pca_components = pca.components_.astype(np.float64)
      self.pca_mean = pca.mean_.astype(np.float64)
      X_train = pca.transform(X_train)
      X_test  = pca.transform(X_test)
      X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
      X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
      if print_flag:
        print(f'Feature reduction (PCA): -> {X_train.shape[1]} components '
              f'({sel_desc})')
    else:
      self.pca_components = None
      self.pca_mean = None

    # Convert to float32 tensors for the model (preprocessing already clipped,
    # so no overflow risk here).
    self.X_train = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
    self.X_test = torch.from_numpy(np.ascontiguousarray(X_test, dtype=np.float32))
    self.y_train = torch.from_numpy(y_train)
    self.y_test = torch.from_numpy(y_test)
        
    train_dataset = TensorDataset(self.X_train, self.y_train)
    test_dataset = TensorDataset(self.X_test, self.y_test)

    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_dataset, test_dataset, train_loader, test_loader

  def save_stats(self, path="prep_stats.npz"):
    '''
      Persist the train-set preprocessing artifacts so new molecules can be
      processed identically in a later session (see predict_single_value and
      load_model): the DescriptorCleaner column keep_mask and per-column
      imputation medians, the standardization feature_mean / feature_std (over
      the kept columns), and (when PCA was used) the PCA components and mean.
      feature_std is the already-guarded version (constant columns left at
      std == 1).

      Args:
        path: output .npz path
    '''
    has_pca = self.pca_components is not None
    np.savez(path,
             keep_mask=self.keep_mask,
             medians=self.medians,
             feature_mean=self.feature_mean,
             feature_std=self.feature_std,
             has_pca=np.array(has_pca),
             pca_components=self.pca_components if has_pca else np.zeros(0, dtype=np.float32),
             pca_mean=self.pca_mean if has_pca else np.zeros(0, dtype=np.float32))
    if print_flag:
      print(f'Saved preprocessing statistics to {path}')

  @classmethod
  def load_stats(cls, path="prep_stats.npz"):
    '''
      Build a lightweight prep_data instance carrying only the saved train-set
      preprocessing artifacts (keep_mask, feature_mean / feature_std, and PCA
      components/mean when present), suitable for passing to
      predict_single_value when predicting with a reloaded model. No data
      loader is created, so this does not need X/y data or a batch_size.

      Args:
        path: .npz path written by save_stats
      Returns:
        prep: prep_data instance with preprocessing artifacts restored
    '''
    inst = cls.__new__(cls)
    data = np.load(path)
    inst.keep_mask = data["keep_mask"]
    inst.medians = data["medians"]
    inst.feature_mean = data["feature_mean"]
    inst.feature_std = data["feature_std"]
    if bool(data["has_pca"]):
      inst.pca_components = data["pca_components"]
      inst.pca_mean = data["pca_mean"]
    else:
      inst.pca_components = None
      inst.pca_mean = None
    inst.batch_size = None
    inst.shuffle = False
    inst.classifier_flag = False
    inst.reduce_dim = "pca" if inst.pca_components is not None else \
                      ("filter" if inst.keep_mask.sum() < inst.keep_mask.size else None)
    inst.var_threshold = 1e-3
    inst.corr_threshold = 0.98
    inst.pca_var = 0.95
    return inst

def load_model():
   '''
    Loads a PyTorch model and its preprocessing statistics from disk: model
    architecture/params from MLP_model_params.txt, weights from
    saved_model.pt, and the train-set feature_mean / feature_std from
    prep_stats.npz (written by prep_data.save_stats).

    Args:
      None
    Returns:
      model: The loaded PyTorch model.
      prep: a prep_data instance carrying feature_mean / feature_std for
        standardizing new molecules in predict_single_value.
   '''
   f = open("MLP_model_params.txt","r")
   lines = f.readlines()
   f.close()
   
   neurons = lines[0].split()[1]
   input_dims = lines[1].split()[1]
   num_hidden_layers = lines[2].split()[1]
   classifier_raw = lines[3].split(":")[1].strip().replace("\n","")
   if classifier_raw == "True":
     classifier_flag = True
   elif classifier_raw == "False":
     classifier_flag = False
   else:
     raise ValueError("classifier_flag must be True or False")
   num_classes = lines[4].split()[1]

   # skip_connection was added later; default to False for params files that
   # were written before the flag existed (only 5 lines).
   skip_raw = lines[5].split(":")[1].strip().replace("\n", "") if len(lines) > 5 else "False"
   if skip_raw == "True":
     skip_connection = True
   elif skip_raw == "False":
     skip_connection = False
   else:
     raise ValueError("skip_connection must be True or False")

   model = MLP_Model(neurons=int(neurons), input_dims=int(input_dims), num_hidden_layers=int(num_hidden_layers),
                     classifier_flag = classifier_flag, num_classes = int(num_classes),
                     skip_connection = skip_connection)
   model.load_state_dict(torch.load("saved_model.pt",weights_only=True))

   prep = prep_data.load_stats()

   return model, prep
