import os
import joblib
from PIL import Image
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import shutil

def buildSampleFromPath(path1, path2):
    h = 250
    l = 250
    S = []
    for filename in os.listdir(path1):
        file_path = os.path.join(path1, filename)

        img = Image.open(file_path).convert("RGB")
        img_resized = resizeImage(img, h, l)
        histo = computeHisto(img_resized)

        image_dict = {
            "name_path": file_path,
            "resized_image": img_resized,
            "X_histo": histo,
            "y_true_class": +1,
            "y_predicted_class": None
        }

        S.append(image_dict)
        img.close()

    for filename in os.listdir(path2):
        file_path = os.path.join(path2, filename)

        img = Image.open(file_path).convert("RGB")
        img_resized = resizeImage(img, h, l)
        histo = computeHisto(img_resized)

        image_dict = {
            "name_path": file_path,
            "resized_image": img_resized,
            "X_histo": histo,
            "y_true_class": -1,
            "y_predicted_class": None
        }

        S.append(image_dict)
        img.close()
    return S

def resizeImage(i, h, l):
    return i.resize((l, h))

def computeHisto(i):
    largeur, hauteur = i.size
    
    # Couper l'image en deux (left, top, right, bottom)
    moitie_haute = i.crop((0, 0, largeur, hauteur // 2))
    moitie_basse = i.crop((0, hauteur // 2, largeur, hauteur))
    
    histo_haut = moitie_haute.histogram()
    histo_bas = moitie_basse.histogram()
    
    return histo_haut + histo_bas

def fitFromHisto(S, algo):
    X = [img["X_histo"] for img in S]
    y = [img["y_true_class"] for img in S]
    if algo["name"] == "GaussianNB":
        model = GaussianNB(**algo["hyper_param"])
    elif algo["name"] == "RandomForest":
        model = RandomForestClassifier(**algo["hyper_param"])
    else:
        raise ValueError("Algorithme pas reconnu")
    model.fit(X, y)
    return model

def predictFromHisto(S, model):
    predictions = []
    for img in S:
        x = img["X_histo"]
        y_pred = model.predict([x])[0]
        img["y_predicted_class"] = y_pred
        predictions.append(y_pred)
    return predictions

def erreurempirique(S):
    y_true = [img["y_true_class"] for img in S]
    y_pred = [img["y_predicted_class"] for img in S]

    accuracy = accuracy_score(y_true, y_pred)
    return 1 - accuracy


def crossValidationError(S, algo, k):
    X = [img["X_histo"] for img in S]
    y = [img["y_true_class"] for img in S]

    if algo["name"] == "GaussianNB":
        model = GaussianNB(**algo["hyper_param"])
    elif algo["name"] == "RandomForest":
        model = RandomForestClassifier(**algo["hyper_param"])
    else:
        raise ValueError("Algorithme non reconnu")

    accuracies = cross_val_score(model, X, y, cv=k)
    return 1 - np.mean(accuracies)

path1 = r"/amuhome/e21226870/projet_aii-1/Init/Mer" #remplacez le chemin par votre propre chemin ou pushez directement les 2 fichiers et remplacez ces 2 paths par les paths adéquats 
path2 = r"/amuhome/e21226870/projet_aii-1/Init/Ailleurs"
S = buildSampleFromPath(path1, path2)
print("Nombre d'images chargees :", len(S))
algo = {"name": "RandomForest", "hyper_param": {"n_estimators": 50, "max_depth": 5, "random_state": 42}}
model = fitFromHisto(S, algo)
predictFromHisto(S, model)

y_true = [img["y_true_class"] for img in S]
y_pred = [img["y_predicted_class"] for img in S]

cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
tp, fn, fp, tn = cm.ravel()

print("--------- Matrice de Confusion ---------")
print(f"Vrais Positifs (Mer bien classée) : {tp}")
print(f"Vrais Négatifs (Ailleurs bien classé) : {tn}")
print(f"Faux Positifs (Ailleurs classé comme Mer) : {fp}")
print(f"Faux Négatifs (Mer classée comme Ailleurs) : {fn}")

# output of the false negatives and positives to manually check
false_neg_dir = "False_Negatives"
false_pos_dir = "False_Positives"
os.makedirs(false_neg_dir, exist_ok=True)
os.makedirs(false_pos_dir, exist_ok=True)

# Mer classée comme Ailleurs
for img in S:
    if img["y_true_class"] == 1 and img["y_predicted_class"] == -1:
        dest = os.path.join(false_neg_dir, os.path.basename(img["name_path"]))
        shutil.copy(img["name_path"], dest)

# Ailleurs classé comme Mer
for img in S:
    if img["y_true_class"] == -1 and img["y_predicted_class"] == 1:
        dest = os.path.join(false_pos_dir, os.path.basename(img["name_path"]))
        shutil.copy(img["name_path"], dest)

ee = erreurempirique(S)
er = crossValidationError(S, algo, k=5)

print(f"Erreur empirique (EE) : {ee:.4f}")
print(f"Erreur réelle (ER) : {er:.4f}")

# enregistrement du model
nom_fichier_modele = "mon_modele_cc2.joblib"
joblib.dump(model, nom_fichier_modele)
print(f"Modèle sauvegardé sous : {nom_fichier_modele}")
