# One-off end-to-end check of the classifier pipeline (not a pytest test --
# run directly). Exercises: prepare_classes -> featurize -> sklearn train/
# grid_search -> evaluate -> inference, then a short MLP + MLP-scan and a tiny
# chemprop classifier, and finally a regression run to prove nothing regressed.
import shutil
import numpy as np

from chemlagent import tools as T

TEST_RUN = "cls_e2e_test"
REG_RUN = "reg_e2e_test"
for rid in (TEST_RUN, REG_RUN):
    shutil.rmtree(f"runs/{rid}", ignore_errors=True)

print("### 1. prepare_chembl_csv")
out = T.prepare_chembl_csv(input_csv="CHEMBL220_bioactives.csv",
                           run_id=TEST_RUN, limit=400)
print(out["n_rows"], "rows")

print("### 2. prepare_classes")
out = T.prepare_classes(TEST_RUN, num_classes=3)
print("classes:", out["num_classes"], out["class_labels"], out["class_counts"])
assert out["num_classes"] == 3

print("### 3. featurize_fingerprints")
out = T.featurize_fingerprints(TEST_RUN, fp_type="ECFP", test_size=0.2)
print("features:", out["n_features"], "task:", out["task"])
feats = np.load("runs/cls_e2e_test/features.npz")
assert set(np.unique(feats["y_test"])) <= {0, 1, 2}

print("### 4. train_model(random_forest)")
out = T.train_model(TEST_RUN, model_type="random_forest")
assert "accuracy" in out, out

print("### 5. grid_search(lightgbm)")
out = T.grid_search(TEST_RUN, "lightgbm",
                    {"n_estimators": [30, 60], "num_leaves": [15, 31]})
assert "best_cv_accuracy" in out, out

print("### 6. train_model(svr)")
out = T.train_model(TEST_RUN, model_type="svr")
assert "accuracy" in out, out

print("### 7. evaluate_model + per-type")
out = T.evaluate_model(TEST_RUN, model_type="random_forest")
assert "mcc" in out and "confusion_matrix" in out, out
print("RF eval:", {k: out[k] for k in
                   ("accuracy", "balanced_accuracy", "mcc", "f1_macro")})
print("labels sample:", out["prediction_labels"][:3])

out = T.evaluate_model(TEST_RUN, model_type="lightgbm")
assert "mcc" in out

print("### 8. run_inference (sklearn, classification)")
smis = ["CCOc1ccc(cc1)C(=O)NC", "O=C(O)c1ccccc1OC(C)C"]
out = T.run_inference(TEST_RUN, smis)
print(out["predicted_labels"])
assert len(out["predicted_class"]) == 2

print("### 9. train_mlp (short)")
out = T.train_mlp(TEST_RUN, epochs=15, patience=4)
assert "accuracy" in out, out

print("### 10. mlp_scan (2 combos)")
out = T.mlp_scan(TEST_RUN,
                 {"neurons": [32, 64], "num_hidden_layers": [0, 1]},
                 epochs=8, patience=3, cv_folds=3)
assert "best_cv_accuracy" in out, out
out = T.evaluate_model(TEST_RUN, model_type="mlp")
assert "mcc" in out
out = T.run_inference_mlp(TEST_RUN, smis)
print("MLP inference labels:", out["predicted_labels"])

print("### 11. chemprop classifier (tiny, CPU)")
out = T.train_chemprop(TEST_RUN, epochs=1, batch_size=16, accelerator="cpu")
assert "accuracy" in out, out
print("chemprop:", {k: out[k] for k in
                    ("accuracy", "mcc", "n_train", "n_test")})
out = T.evaluate_model(TEST_RUN, model_type="chemprop")
assert "mcc" in out
out = T.run_inference_chemprop(TEST_RUN, smis, accelerator="cpu")
print("chemprop inference labels:", out["predicted_labels"])

print("### 12. regression run unchanged")
T.prepare_chembl_csv(input_csv="CHEMBL220_bioactives.csv",
                     run_id=REG_RUN, limit=400)
T.featurize_fingerprints(REG_RUN, fp_type="ECFP", test_size=0.2)
out = T.train_model(REG_RUN, model_type="lightgbm")
assert "r2" in out, out
out = T.evaluate_model(REG_RUN)
assert "r2" in out and "mcc" not in out
out = T.run_inference(REG_RUN, smis)
assert isinstance(out["predictions"][0], float)

print("ALL E2E CHECKS PASSED")