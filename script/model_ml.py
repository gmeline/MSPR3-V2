import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
os.makedirs('../MSPR/images', exist_ok=True)
print("=" * 80)
print("MODÈLE DE PRÉDICTION - ÉLECTIONS PRÉSIDENTIELLES 2022")
print("Hauts-de-France")
print("=" * 80)

# ============================================================================
# Chargement des données
# ============================================================================

df = pd.read_csv('data_clean/table_finale_famille.csv')
print(f"\nDataset chargé : {len(df)} lignes")
print(f"Colonnes : {df.columns.tolist()}")
print("\n Aperçu des données :")
print(df.head())

# ============================================================================
# Nettouyage et preparation
# ============================================================================

if 'autre' in df['famille_politique'].values:
    df = df[df['famille_politique'] != 'autre']
    print(f"\nLignes 'autre' supprimées. Nouveau total : {len(df)}")

print("\nValeurs manquantes :")
print(df.isnull().sum())

if df['taux_chomage'].isnull().any():
    df['taux_chomage'].fillna(df['taux_chomage'].mean(), inplace=True)
    print("Valeurs manquantes du taux_chomage remplacées par la moyenne")

# ============================================================================
# Aggregation par departement afin d'idnetifier le gagnant
# ============================================================================

print("\n" + "=" * 80)
print("AGRÉGATION DES DONNÉES PAR DÉPARTEMENT")
print("=" * 80)
votes_par_famille = df.groupby(['code_insee', 'departement', 'famille_politique'])['voix'].sum().reset_index()
print(f"\nVotes agrégés : {len(votes_par_famille)} lignes")
idx_max = votes_par_famille.groupby('code_insee')['voix'].idxmax()
gagnants = votes_par_famille.loc[idx_max][['code_insee', 'departement', 'famille_politique']]
gagnants.rename(columns={'famille_politique': 'famille_gagnante'}, inplace=True)

print("\n🏆 Famille politique gagnante par département :")
print(gagnants)

# ============================================================================
# Creation des features
# ============================================================================

colonnes_stats = []
for col in ['Inscrits', 'Abstentions', 'Votants', 'Blancs', 'Nuls', 'Exprimés']:
    if col in df.columns:
        colonnes_stats.append(col)
print(f"\nColonnes statistiques détectées : {colonnes_stats}")
agg_dict = {'taux_chomage': 'mean', 'voix': 'sum'}
for col in colonnes_stats:
    agg_dict[col] = 'sum'
features_dept = df.groupby(['code_insee', 'departement']).agg(agg_dict).reset_index()
if 'Abstentions' in features_dept.columns and 'Inscrits' in features_dept.columns:
    features_dept['taux_abstention'] = (features_dept['Abstentions'] / features_dept['Inscrits']) * 100
if 'Votants' in features_dept.columns and 'Inscrits' in features_dept.columns:
    features_dept['taux_participation'] = (features_dept['Votants'] / features_dept['Inscrits']) * 100
if 'Blancs' in features_dept.columns and 'Votants' in features_dept.columns:
    features_dept['taux_blancs'] = (features_dept['Blancs'] / features_dept['Votants']) * 100
ml_dataset = features_dept.merge(gagnants, on=['code_insee', 'departement'])
print("\nDataset ML créé :")
print(ml_dataset.head())
print(f"\n{len(ml_dataset)} départements dans le dataset ML")

# ============================================================================
# Encodage de la sortie
# ============================================================================

le = LabelEncoder()
ml_dataset['target'] = le.fit_transform(ml_dataset['famille_gagnante'])

print("\nEncodage des familles politiques :")
for i, famille in enumerate(le.classes_):
    print(f"  {i} = {famille}")

# ============================================================================
# Selection des features
# ============================================================================

features_possibles = ['taux_chomage', 'taux_abstention', 'taux_participation', 'taux_blancs']
features_cols = [f for f in features_possibles if f in ml_dataset.columns]
print(f"\nFeatures sélectionnées pour le ML : {features_cols}")
if len(features_cols) == 0:
    print("ERR : Aucune feature disponible pour le ML")
    exit(1)
X = ml_dataset[features_cols]
y = ml_dataset['target']
print(f"\nX shape : {X.shape}")
print(f"y shape : {y.shape}")

# ============================================================================
# Entrainement/test
# ============================================================================

if len(X) < 4:
    print(f"WARN : Seulement {len(X)} départements, split difficile")
    test_size = 0.25 
else:
    test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y if len(X) >= 10 else None
)

print(f"\nSplit des données :")
print(f"  Train : {len(X_train)} observations ({100-test_size*100:.0f}%)")
print(f"  Test  : {len(X_test)} observations ({test_size*100:.0f}%)")

# ============================================================================
# Entrainement des modeles
# ============================================================================
models = {
    'Régression Logistique': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
}
results = {}
print("\n" + "=" * 80)
print("ENTRAÎNEMENT DES MODÈLES")
print("=" * 80)
for name, model in models.items():
    print(f"\n🤖 Modèle : {name}")
    print("-" * 40)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"  Accuracy (train) : {acc_train:.3f}")
    print(f"  Accuracy (test)  : {acc_test:.3f}")
    print("\n  Classification Report (test) :")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_, zero_division=0))
    results[name] = {
        'model': model,
        'acc_train': acc_train,
        'acc_test': acc_test,
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train
    }

# ============================================================================
# Visualisation
# ============================================================================

print("\n" + "=" * 80)
print("GÉNÉRATION DES VISUALISATIONS")
print("=" * 80)

# ============================================================================
# Visu 1 : Comparaison des accuracy
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(results.keys())
train_accs = [results[m]['acc_train'] for m in model_names]
test_accs = [results[m]['acc_test'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, train_accs, width, label='Train', alpha=0.8, color='#3498db')
bars2 = ax.bar(x + width/2, test_accs, width, label='Test', alpha=0.8, color='#e74c3c')

ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Comparaison des performances des modèles\nÉlections Présidentielles 2022 - Hauts-de-France', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.legend(fontsize=10)
ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Seuil minimum (0.5)', alpha=0.7)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.2f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../MSPR/images/ml_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("ml_accuracy_comparison.png")
plt.close()

# ============================================================================
# Visu 2 : Matrice de confusion
# ============================================================================
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

if n_models == 1:
    axes = [axes]

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred_test'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_, 
                ax=axes[idx], cbar_kws={'label': 'Nombre de prédictions'})
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Vraie classe', fontsize=10)
    axes[idx].set_xlabel('Classe prédite', fontsize=10)

plt.suptitle('Matrices de confusion - Test Set', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../MSPR/images/ml_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("ml_confusion_matrices.png")
plt.close()

# ============================================================================
# Visu 3 : Random Forest
# ============================================================================
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': features_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
bars = ax.barh(feature_importance['feature'], feature_importance['importance'], color=colors, alpha=0.8)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Importance des variables prédictives\nRandom Forest - Hauts-de-France 2022', 
             fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
            f'{width:.3f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('../MSPR/images/ml_feature_importance.png', dpi=300, bbox_inches='tight')
print("ml_feature_importance.png")
plt.close()

# ============================================================================
# Visu 4 : régression logistique
# ============================================================================

lr_model = results['Régression Logistique']['model']
y_score = lr_model.predict_proba(X_test)

y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
            label=f'{le.classes_[i]} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.50)', alpha=0.5)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Taux de Faux Positifs', fontsize=12, fontweight='bold')
ax.set_ylabel('Taux de Vrais Positifs', fontsize=12, fontweight='bold')
ax.set_title('Courbes ROC - Régression Logistique\nÉlections Présidentielles 2022 - Hauts-de-France',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../MSPR/images/ml_roc_curves.png', dpi=300, bbox_inches='tight')
print("ml_roc_curves.png")
plt.close()

# ============================================================================
# Enregistrement des résultat
# ============================================================================

os.makedirs('../data_final', exist_ok=True)

ml_dataset.to_csv('../data_final/ml_dataset_prepared.csv', index=False)
print("\nDataset ML sauvegardé : data_final/ml_dataset_prepared.csv")

best_model_name = max(results.items(), key=lambda x: x[1]['acc_test'])[0]
best_model = results[best_model_name]['model']

predictions_df = ml_dataset.copy()
predictions_df['prediction'] = best_model.predict(X)
predictions_df['prediction_famille'] = le.inverse_transform(predictions_df['prediction'])
proba = best_model.predict_proba(X)
for i, classe in enumerate(le.classes_):
    predictions_df[f'proba_{classe}'] = proba[:, i]

predictions_df.to_csv('../data_final/ml_predictions.csv', index=False)
print("Prédictions sauvegardées : data_final/ml_predictions.csv")

# ============================================================================
# Conclusion
# ============================================================================

print("\n" + "=" * 80)
print("RÉSUMÉ DES RÉSULTATS")
print("=" * 80)

best_acc = results[best_model_name]['acc_test']

print(f"\n🏆 Meilleur modèle : {best_model_name}")
print(f"   Accuracy (train) : {results[best_model_name]['acc_train']:.3f}")
print(f"   Accuracy (test)  : {best_acc:.3f}")
print(f"   {'VALIDÉ' if best_acc > 0.5 else 'NON VALIDÉ'} (seuil > 0.5)")

print("\nFichiers générés dans le dossier images/ :")
print("  - ml_accuracy_comparison.png")
print("  - ml_confusion_matrices.png")
print("  - ml_feature_importance.png")
print("  - ml_roc_curves.png")

print("\nFichiers de données générés dans data_final/ :")
print("  - ml_dataset_prepared.csv")
print("  - ml_predictions.csv")

print("\nTRAITEMENT TERMINÉ !")
print("=" * 80)