import pandas as pd
import requests
import os
import json
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
import numpy as np

try:
  from chembl_webresource_client.new_client import new_client
  chembl_flag = True
except:
  print("Failed to import chembl_webresource_client. Some functionality may be limited.")
  chembl_flag = False

# Module-level print flag - set from modrag.py
print_flag = False

# Distance (Angstrom) under which a co-crystallized molecule is reported as
# "close" to the docked ligand (minimum atom-to-atom distance). Mirrors
# vina_dock.NEAR_LIGAND_CUTOFF so "near a known ligand" means the same thing
# in the post-hoc check and the blind-dock proximity fallback.
NEARBY_DISTANCE_CUTOFF = 5.0

def uniprot_node(protein_names: list[str], human_flag: bool = False) -> (list[str], str):
  '''
    This tool takes in the user requested protein and searches UNIPROT for matches.
    It returns a string scontaining the protein ID, gene name, organism, and protein name.
      Args:
        query_protein: the name of the protein to search for.

      Returns:
        total_ids: a list of UNIPROT IDs for the given protein names.
        protein_string: a string containing the protein ID, gene name, organism, and protein name.

  '''
  print("UNIPROT tool")
  print('===================================================')

  total_ids = []
  protein_string = ''

  for protein_name in protein_names:
    try:
      url = f'https://rest.uniprot.org/uniprotkb/search?query={protein_name}&format=tsv'
      response = requests.get(url).text

      f = open(f"../scratch/{protein_name}_uniprot_ids.tsv", "w")
      f.write(response)
      f.close()

      prot_df_raw = pd.read_csv(f'../scratch/{protein_name}_uniprot_ids.tsv', sep='\t')
      if human_flag:
        prot_df = prot_df_raw[prot_df_raw['Organism'] == "Homo sapiens (Human)"]
        if print_flag:
            print(f"Found {len(prot_df)} Human proteins out of {len(prot_df_raw)} total proteins")
      else:
        prot_df = prot_df_raw

      prot_ids = prot_df['Entry'].tolist()
      genes = prot_df['Gene Names'].tolist()
      organisms = prot_df['Organism'].tolist()
      names = prot_df['Protein names'].tolist()

      sub_ids = []
      for id, gene, organism, name in zip(prot_ids, genes, organisms, names):
        protein_string += f'Protein {protein_name}, ID: {id}, Gene: {gene}, Organism: {organism}, Name: {name}\n'
        sub_ids.append(id)

      protein_string += '==========================================================================================\n'
      total_ids.append(sub_ids)
    except:
      protein_string += f'No proteins found for {protein_name}'
      protein_string += '==========================================================================================\n'
      total_ids.append([])

  return total_ids, protein_string, None

def listbioactives_node(up_ids_list: list[str]) -> (list[int], list[str], str):
  '''
    Accepts a UNIPROT ID and searches for bioactive molecules
      Args: 
        up_ids_list: the UNIPROT IDs of the proteins to search for.
      Returns:
        total_bioacts_list: a list of the number of bioactive molecules for each protein
        total_chembl_ids_list: a list of the ChEMBL IDs for each protein
        bioact_string: a string containing the results of the search.
  '''
  print("List bioactives tool")
  print('===================================================')
  if not chembl_flag:
    print("ChEMBL client not available at this time")
    return [], "ChEMBL client not available.", []

  total_bioacts_list = []
  total_chembl_ids_list = []
  bioact_string = ''

  for up_id in up_ids_list:

    targets = new_client.target
    bioact = new_client.activity

    try:
      target_info = targets.get(target_components__accession=up_id).only("target_chembl_id","organism", "pref_name", "target_type")
      target_info = pd.DataFrame.from_records(target_info)
      if print_flag:
          print(target_info)
      if len(target_info) > 0:
        if print_flag:
            print(f"Found info for Uniprot ID: {up_id}")

      chembl_ids = target_info['target_chembl_id'].tolist()

      chembl_ids = list(set(chembl_ids))
      if print_flag:
          print(f"Found {len(chembl_ids)} unique ChEMBL IDs")

      len_all_bioacts = []
      for chembl_id in chembl_ids:
        bioact_chosen = bioact.filter(target_chembl_id=chembl_id, type="IC50", relation="=").only(
            "molecule_chembl_id",
            "type",
            "standard_units",
            "relation",
            "standard_value",
        )
        len_this_bioacts = len(bioact_chosen)
        len_all_bioacts.append(len_this_bioacts)
        bioact_string += f"For Uniprot {up_id}: length of Bioactivities for ChEMBL ID {chembl_id}: {len_this_bioacts}\n"

      bioact_string += f'================================================================================================\n'
      total_chembl_ids_list.append(chembl_ids)
      total_bioacts_list.append(len_all_bioacts)

    except:
      bioact_string += f'No bioactives found for Uniprot {up_id}\n'
      bioact_string += f'================================================================================================\n'
      total_chembl_ids_list.append([])
      total_bioacts_list.append([])

  return total_bioacts_list, bioact_string, total_chembl_ids_list

def getbioactives_node(chembl_ids_list: list[str]) -> (list[str], str):
  '''
    Accepts a Chembl ID and get all bioactives molecule SMILES and IC50s for that ID
      Args:
        chembl_id: the chembl ID to query
      Returns:
        bioactives_list: a list of the bioactive molecules for each chembl ID
        bioactives_string: a string containing the results of the search.
        bioactives_images: a list of images for each bioactive molecule.
  '''
  print("Get bioactives tool")
  print('===================================================')
  if not chembl_flag:
    print("ChEMBL client not available at this time")
    return [], "ChEMBL client not available.", None

  bioactives_list = []
  bioactives_images = []
  bioactives_string = ''

  for chembl_id in chembl_ids_list:
    try:
      #check if f'{chembl_id}_bioactives.csv' exists
      chembl_id = chembl_id.upper()
      if os.path.exists(f'../scratch/{chembl_id}_bioactives.csv'):
        if print_flag:
            print(f'Found {chembl_id}_bioactives.csv')
        total_bioact_df = pd.read_csv(f'../scratch/{chembl_id}_bioactives.csv')
        if print_flag:
            print(f"number of records: {len(total_bioact_df)}")
      else:

        compounds = new_client.molecule
        bioact = new_client.activity

        bioact_chosen = bioact.filter(target_chembl_id=chembl_id, type="IC50", relation="=").only(
            "molecule_chembl_id",
            "type",
            "standard_units",
            "relation",
            "standard_value",
        )

        chembl_ids = []
        ic50s = []
        for record in bioact_chosen:
            if record["standard_units"] == 'nM':
                chembl_ids.append(record["molecule_chembl_id"])
                ic50s.append(float(record["standard_value"]))

        bioact_dict = {'chembl_ids' : chembl_ids, 'IC50s': ic50s}
        bioact_df = pd.DataFrame.from_dict(bioact_dict)
        bioact_df.drop_duplicates(subset=["chembl_ids"], keep= "last")
        if print_flag:
            print(f"Number of records: {len(bioact_df)}")
            print(bioact_df.shape)

        compounds_provider = compounds.filter(molecule_chembl_id__in=bioact_df["chembl_ids"].to_list()).only(
            "molecule_chembl_id",
            "molecule_structures"
        )

        cids_list = []
        smiles_list = []

        for record in compounds_provider:
            cid = record['molecule_chembl_id']
            cids_list.append(cid)

            if record['molecule_structures']:
                if record['molecule_structures']['canonical_smiles']:
                    smile = record['molecule_structures']['canonical_smiles']
                else:
                    if print_flag:
                        print("No canonical smiles")
                    smile = None
            else:
                if print_flag:
                    print('no structures')
                smile = None
            smiles_list.append(smile)

        new_dict = {'SMILES': smiles_list, 'chembl_ids_2': cids_list}
        new_df = pd.DataFrame.from_dict(new_dict)

        total_bioact_df = pd.merge(bioact_df, new_df, left_on='chembl_ids', right_on='chembl_ids_2')
        if print_flag:
            print(f"number of records: {len(total_bioact_df)}")

        total_bioact_df.drop_duplicates(subset=["chembl_ids"], keep= "last")
        if print_flag:
            print(f"number of records after removing duplicates: {len(total_bioact_df)}")

        total_bioact_df.dropna(axis=0, how='any', inplace=True)
        total_bioact_df.drop(["chembl_ids_2"],axis=1,inplace=True)
        if print_flag:
            print(f"number of records after dropping Null values: {len(total_bioact_df)}")

        total_bioact_df.sort_values(by=["IC50s"],inplace=True)

        if len(total_bioact_df) > 0:
          total_bioact_df.to_csv(f'../scratch/{chembl_id}_bioactives.csv')

      limit = 50
      if len(total_bioact_df) > limit:
        total_bioact_df = total_bioact_df.iloc[:limit]

      bioact_tuple_list = []
      bioactives_string += f'Results for top bioactivity (IC50 value) for molecules in ChEMBL ID: {chembl_id}. \n'
      for smile, ic50 in zip(total_bioact_df['SMILES'], total_bioact_df['IC50s']):
        bioactives_string += f'Molecule SMILES: {smile}, IC50 (nM): {ic50}\n'
        bioact_tuple_list.append((smile, ic50))
      bioactives_string += f'=========================================================================================\n'

      mols = [Chem.MolFromSmiles(smile) for smile in total_bioact_df['SMILES'].to_list()]
      legends = [f'IC50: {ic50}' for ic50 in total_bioact_df['IC50s'].to_list()]
      img = MolsToGridImage(mols, molsPerRow=5, legends=legends, subImgSize=(200,200))
      # Save image to chat location
      if not os.path.exists('../images'):
        os.makedirs('../images')
      img.save('../images/chat_image.png')
      bioactives_images.append(img)
      bioactives_list.append(bioact_tuple_list)
    except: 
      bioactives_list.append([])
      bioactives_string += f'No bioactives found for ChEMBL ID: {chembl_id}\n'
      bioactives_string += f'=========================================================================================\n'
      bioactives_images.append(None)

  return bioactives_list, bioactives_string, bioactives_images

def target_node(search_descriptors: list[str]):
  '''
  Accepts a disease name and searches Open Targets for associated targets

  Args:
    search_descriptor (str): Disease name

  Returns:
    targets_list (list): List of targets
    targets_string (str): String of targets
    None
  '''
  print("Open Targets tool")
  print('===================================================')
  base_url = "https://api.platform.opentargets.org/api/v4/graphql"

  disease_query_string = """
    query searchEntity($queryString: String!) {
      search(queryString: $queryString){
        total
        hits  {
          id
          entity
          description
        }
      }
    }
  """

  target_query_string = """
    query associatedTargets($efo_id: String!) {
      disease(efoId: $efo_id) {
        id
        name
        associatedTargets {
          count
          rows {
            target {
              id
              approvedSymbol
            }
            score
          }
        }
      }
    }
  """
  total_targets_list = []
  total_targets_string = ''

  for search_descriptor in search_descriptors:

    variables = {"queryString": search_descriptor}
    r = requests.post(base_url, json={"query": disease_query_string, "variables": variables})

    disease_list = []
    targets_list = []

    if r.status_code == 200:
      api_response = json.loads(r.text)
      if len(api_response['data']['search']['hits']) > 0:
        for hit in api_response['data']['search']['hits']:
          if hit['entity'] == 'disease':
            disease_list.append(hit['id'])
    else:
      if print_flag:
          print('Could not find results.')

    if len(disease_list) > 0:
      q = requests.post(base_url, json={"query": target_query_string, "variables": {"efo_id": disease_list[0]}})
      if q.status_code == 200:
        api_response = json.loads(q.text)
        for target in api_response['data']['disease']['associatedTargets']['rows']:
          targets_list.append(target['target']['approvedSymbol'])

    targets_string = f'Possible targets for {search_descriptor} include: \n'
    if len(targets_list) > 0:
      for i, target in enumerate(targets_list):
        targets_string += f'{i+1}. {target}\n'
    else:
      targets_string = f'No targets found for {search_descriptor}'
    
    total_targets_list.append(targets_list)
    total_targets_string += targets_string

  return total_targets_list, total_targets_string, None

def predict_node(smiles_list_in: list[str], chembl_id: str) -> (list[float],str,list):
  '''
    uses the current_bioactives.csv file from the get_bioactives node to fit the
    Light GBM model and predict the IC50 for the current smiles.
      Args:
        smiles_list: the SMILES strings of the molecules to predict
        chembl_id: the chembl ID to query
      Returns:
        preds: a list of predicted IC50 values for the input SMILES
        preds_string: a string containing the predicted IC50 values for the input SMILES
        preds_images: a list of images for each predicted molecule (currently not implemented)
  '''
  print("Predict Tool")
  print('===================================================')

  # if f'{chembl_id}_bioactives.csv' does not exist, call the bioactives node
  if not os.path.exists(f'../scratch/{chembl_id}_bioactives.csv'):
    _, _, _ = getbioactives_node([chembl_id])
  
  try:
    chembl_id = chembl_id.upper()
    df = pd.read_csv(f'../scratch/{chembl_id}_bioactives.csv')
    #if length of the dataframe is over 2000, take a random sample of 2000 points
    if len(df) > 2000:
      df = df.sample(n=2000, random_state=42)

    y_raw = df["IC50s"].to_list()
    smiles_list = df["SMILES"].to_list()
    ions_to_clean = ["[Na+].",".[Na+]","[Cl-].",".[Cl-]","[K+].",".[K+]"]
    Xa = []
    y = []
    for smile, value in zip(smiles_list, y_raw):
      for ion in ions_to_clean:
        smile = smile.replace(ion,"")
      # IC50 <= 0 is physically meaningless (missing/placeholder value); log10 is
      # undefined there and would inject -inf into the training targets, so skip
      # those rows rather than emitting a divide-by-zero warning.
      if value is None or value <= 0:
        if print_flag:
          print(f"Skipping molecule with non-positive IC50 ({value}): {smile}")
        continue
      y.append(np.log10(value))
      Xa.append(smile)

    if print_flag:
      print(f"Number of molecules: {len(Xa)}")

   
    f, y, Xa = rdkit_featurize(Xa, y)
    f = np.array(f)

    nan_indicies = np.isnan(f)
    bad_rows = []
    for i, row in enumerate(nan_indicies):
        for item in row:
            if item == True:
                if i not in bad_rows:
                    if print_flag:
                      print(f"Row {i} has a NaN.")
                    bad_rows.append(i)

    if print_flag:
      print(f"Old dimensions are: {f.shape}.")

    for j,i in enumerate(bad_rows):
        k=i-j
        f = np.delete(f,k,axis=0)
        y = np.delete(y,k,axis=0)
        Xa = np.delete(Xa,k,axis=0)
        if print_flag:
          print(f"Deleting row {k} from arrays.")

    if print_flag:
      print(f"New dimensions are: {f.shape}")
    if f.shape[0] != len(y) or f.shape[0] != len(Xa):
      raise ValueError("Number of rows in X and y do not match.")

    # Create feature names for the model
    feature_names = [f"descriptor_{i}" for i in range(f.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to DataFrames with feature names
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    # n_jobs=1 forces single-threaded LightGBM. Inside the modrag process, torch
    # (Metal/MPS) and the meeko/autodock-vina native libs are already loaded, and
    # LightGBM's default OpenMP parallel region (n_jobs=-1) SIGSEGVs (exit 139) when
    # it spawns OMP threads against that already-initialized runtime — predict_node
    # dies right after "New dimensions are: ...". Disabling OpenMP parallelism
    # avoids the conflict; on ~2k rows x 217 descriptors the fit is sub-second
    # anyway, so there is no real cost. Do NOT set this back to -1. See memory note
    # gpt-finetune-chembl220-crash for the same class of native-lib/Metal crash.
    model = LGBMRegressor(metric='rmse', max_depth = 50, verbose = -1, num_leaves = 31,
                          feature_fraction = 0.8, min_data_in_leaf = 20, n_jobs=1)
    modelname = "LightGBM Regressor"
    model.fit(X_train, y_train)

    train_score = model.score(X_train,y_train)
    if print_flag:
      print(f"score for training set: {train_score:.3f}")

    valid_score = model.score(X_test, y_test)
    if print_flag:
      print(f"score for validation set: {valid_score:.3f}")
  except:
    return [], 'Model training failed, unable to predict.', None

  preds = []
  preds_string = ''

  for smiles in smiles_list_in:
    if print_flag:
      print(f"in predict node, smiles: {smiles}")
    idx = 0
    try:
      for ion in ions_to_clean:
        smiles = smiles.replace(ion,"")
      test_feat, _, _ = rdkit_featurize([smiles], None)
      test_feat = scaler.transform(test_feat)
      # Convert to DataFrame with same feature names used during training
      test_feat = pd.DataFrame(test_feat, columns=feature_names)
      prediction = model.predict(test_feat)
      test_ic50 = 10**(prediction[0])
      if print_flag:
        print(f"Predicted IC50 for {smiles}: {test_ic50}")
      preds_string += f"The predicted IC50 value for {smiles} is : {test_ic50:.3f} nM.\n"
      
      preds.append(test_ic50)
      idx+=1
    except:
      preds.append(None)
      preds_string += f"The prediction for {smiles} failed.\n"

  preds_string += f"The Bioactive data was fitted with the LightGMB model, using RDKit descriptors. The training score \
was {train_score:.3f} and the testing score was {valid_score:.3f}. "

  return preds, preds_string, None
