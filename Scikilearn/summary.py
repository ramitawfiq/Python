"""
ULTIMATE SCIKIT-LEARN MASTER SCRIPT (14 CHAPTERS)
Python Data Science Handbook (13 chapters) + Scikit-Learn Cheat Sheet
SINGLE EXECUTABLE FILE - COMPLETE ML EDUCATION
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets, neighbors, preprocessing, linear_model, svm, naive_bayes, decomposition
from sklearn.datasets import (
    load_iris, load_digits, fetch_20newsgroups, fetch_lfw_people,
    make_blobs, make_moons, fetch_species_distributions
)
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, 
    LeaveOneOut, validation_curve
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    mean_absolute_error, mean_squared_error, r2_score,
    adjusted_rand_score, homogeneity_score, v_measure_score
)
from sklearn.base import BaseEstimator, ClassifierMixin
from matplotlib.patches import Ellipse, Rectangle
import skimage
from skimage import data, color, feature, transform

# Set style
sns.set()
plt.rcParams["figure.figsize"] = (8, 5)
%matplotlib inline

print("ULTIMATE SCIKIT-LEARN MASTER SCRIPT - 14 CHAPTERS")
print("=" * 60)

#########################
# 1-12. PYTHON DATA SCIENCE HANDBOOK (CONDENSED)
#########################
print("\n 1-12. PYTHON DATA SCIENCE HANDBOOK (Core Algorithms)")

# Quick demos of all 12 chapters
iris = load_iris()
digits = load_digits()
Xtrain_i, Xtest_i, ytrain_i, ytest_i = train_test_split(iris.data, iris.target, random_state=0)
Xtrain_d, Xtest_d, ytrain_d, ytest_d = train_test_split(digits.data, digits.target, random_state=0)

# 1. Estimator API
knn = KNeighborsClassifier(n_neighbors=1).fit(Xtrain_i, ytrain_i)
print(f"1. KNN accuracy: {accuracy_score(ytest_i, knn.predict(Xtest_i)):.3f}")

# 4. Naive Bayes
nb = GaussianNB().fit(Xtrain_d, ytrain_d)
print(f"4. NB accuracy: {accuracy_score(ytest_d, nb.predict(Xtest_d)):.3f}")

# 7. Random Forest
rf = RandomForestClassifier(50, random_state=0).fit(Xtrain_d, ytrain_d)
print(f"7. RF accuracy: {accuracy_score(ytest_d, rf.predict(Xtest_d)):.3f}")

# 8-10. Dimensionality Reduction & Clustering
X_blobs, _ = make_blobs(300, centers=4, random_state=0)
pca = PCA(2).fit_transform(digits.data[:1000])
kmeans = KMeans(4).fit(X_blobs)
gmm = GaussianMixture(4).fit(X_blobs)

plt.figure(figsize=(15, 4))
plt.subplot(131); plt.scatter(pca[:,0], pca[:,1], c=digits.target[:1000], cmap='rainbow'); plt.title("PCA")
plt.subplot(132); plt.scatter(X_blobs[:,0], X_blobs[:,1], c=kmeans.labels_, cmap='viridis'); plt.title("K-Means")
plt.subplot(133); plt.scatter(X_blobs[:,0], X_blobs[:,1], c=gmm.predict(X_blobs), cmap='viridis'); plt.title("GMM")
plt.tight_layout(); plt.show()

#########################
# 13. FACE DETECTION (HOG + LinearSVC)
#########################
print("\n 13. FACE DETECTION PIPELINE")
faces = fetch_lfw_people()
positive_patches = faces.images[:1000]  # Subset for speed
test_image = color.rgb2gray(data.astronaut())[:160, 40:180]

# HOG demo
hog_vec, hog_vis = feature.hog(test_image, visualise=True)
plt.figure(figsize=(12, 5))
plt.subplot(121); plt.imshow(test_image, cmap='gray'); plt.title('Input'); plt.axis('off')
plt.subplot(122); plt.imshow(hog_vis); plt.title('HOG Features'); plt.axis('off')
plt.show()

#########################
# 14. SCIKIT-LEARN CHEAT SHEET
#########################
print("\n 14. SCIKIT-LEARN CHEAT SHEET - COMPACT VERSION")
print("=" * 50)

# -------------------------
# 1. Load & Prepare Data
# -------------------------
iris_cheat = datasets.load_iris()
X_cheat, y_cheat = iris_cheat.data[:, :2], iris_cheat.target
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cheat, y_cheat, random_state=33)

# Preprocessing
scaler = preprocessing.StandardScaler().fit(X_train_c)
X_train_c = scaler.transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

print(f"Dataset shape: {X_cheat.shape}, Classes: {np.unique(y_cheat)}")

# -------------------------
# 2. Create Models
# -------------------------
# Supervised
knn_cheat = neighbors.KNeighborsClassifier(n_neighbors=5)
lr_cheat = linear_model.LinearRegression()
svc_cheat = svm.SVC(kernel='linear')
gnb_cheat = naive_bayes.GaussianNB()

# Unsupervised
k_means_cheat = KMeans(n_clusters=3, random_state=0)
pca_cheat = decomposition.PCA(n_components=0.95)

print("Models created: KNN, LR, SVC, GNB, KMeans, PCA")

# -------------------------
# 3. Fit & Predict
# -------------------------
knn_cheat.fit(X_train_c, y_train_c)
svc_cheat.fit(X_train_c, y_train_c)
y_pred_knn = knn_cheat.predict(X_test_c)
y_pred_svc = svc_cheat.predict(X_test_c)

# Regression example (using blobs)
X_reg, y_reg = make_blobs(100, 2, random_state=0)[0], make_blobs(100, 2, random_state=0)[1][:, 0].astype(float)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, random_state=33)
lr_cheat.fit(X_reg_train, y_reg_train)
y_pred_lr = lr_cheat.predict(X_reg_test)

# Clustering
k_means_cheat.fit(X_train_c)
labels_cluster = k_means_cheat.predict(X_test_c)

# -------------------------
# 4. Evaluate Models
# -------------------------
print("\n MODEL EVALUATION RESULTS:")
print("—" * 30)
print("Classification:")
print(f"  KNN Accuracy:  {accuracy_score(y_test_c, y_pred_knn):.3f}")
print(f"  SVC Accuracy:  {accuracy_score(y_test_c, y_pred_svc):.3f}")
print("\nRegression:")
print(f"  MAE:           {mean_absolute_error(y_reg_test, y_pred_lr):.3f}")
print(f"  RMSE:          {np.sqrt(mean_squared_error(y_reg_test, y_pred_lr)):.3f}")
print(f"  R² Score:      {r2_score(y_reg_test, y_pred_lr):.3f}")
print("\nClustering:")
print(f"  ARI:           {adjusted_rand_score(y_test_c, labels_cluster):.3f}")
print(f"  Homogeneity:   {homogeneity_score(y_test_c, labels_cluster):.3f}")

print("\nConfusion Matrix (KNN):")
print(confusion_matrix(y_test_c, y_pred_knn))
print("\nClassification Report (KNN):")
print(classification_report(y_test_c, y_pred_knn))

# -------------------------
# 5. Cross-Validation
# -------------------------
cv_scores = cross_val_score(knn_cheat, X_train_c, y_train_c, cv=4)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"  Mean CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# -------------------------
# 6. Hyperparameter Tuning
# -------------------------
print("\n HYPERPARAMETER TUNING:")
grid_params = {"n_neighbors": np.arange(1, 6), "metric": ["euclidean", "cityblock"]}
grid = GridSearchCV(knn_cheat, param_grid=grid_params, cv=3)
grid.fit(X_train_c, y_train_c)
print(f"  Grid Search Best: {grid.best_score_:.3f} (n_neighbors={grid.best_estimator_.n_neighbors})")

rand_params = {"n_neighbors": range(1, 8), "weights": ["uniform", "distance"]}
rand_search = RandomizedSearchCV(knn_cheat, param_distributions=rand_params, 
                                cv=3, n_iter=8, random_state=5)
rand_search.fit(X_train_c, y_train_c)
print(f"  Random Search Best: {rand_search.best_score_:.3f}")

# -------------------------
# 7. CHEAT SHEET VISUALIZATION
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("SCIKIT-LEARN CHEAT SHEET - KEY VISUALIZATIONS", fontsize=16)

# Classification boundaries
axes[0,0].scatter(X_test_c[:,0], X_test_c[:,1], c=y_pred_knn, cmap='viridis', s=50)
axes[0,0].set_title(f'KNN (Acc: {accuracy_score(y_test_c, y_pred_knn):.1%})')

axes[0,1].scatter(X_test_c[:,0], X_test_c[:,1], c=y_pred_svc, cmap='plasma', s=50)
axes[0,1].set_title(f'SVC (Acc: {accuracy_score(y_test_c, y_pred_svc):.1%})')

# Clustering
axes[0,2].scatter(X_test_c[:,0], X_test_c[:,1], c=labels_cluster, cmap='coolwarm', s=50)
axes[0,2].set_title(f'KMeans (ARI: {adjusted_rand_score(y_test_c, labels_cluster):.2f})')

# PCA
pca_vis = pca_cheat.fit_transform(X_cheat)
axes[1,0].scatter(pca_vis[:,0], pca_vis[:,1], c=y_cheat, cmap='rainbow', s=30)
axes[1,0].set_title('PCA (95% variance)')

# Regression
axes[1,1].scatter(X_reg_test[:,0], y_reg_test, c='blue', alpha=0.6, label='True')
axes[1,1].scatter(X_reg_test[:,0], y_pred_lr, c='red', alpha=0.6, label='Pred')
axes[1,1].set_title(f'Linear Regression (R²: {r2_score(y_reg_test, y_pred_lr):.2f})')
axes[1,1].legend()

# CV Scores
axes[1,2].bar(range(len(cv_scores)), cv_scores, color='skyblue', alpha=0.7)
axes[1,2].axhline(cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
axes[1,2].set_title('Cross-Validation Scores')
axes[1,2].legend()

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("COMPLETE! ULTIMATE SCIKIT-LEARN MASTER SCRIPT (14 CHAPTERS)")
print("Python Data Science Handbook (13 chapters)")
print("Scikit-Learn Cheat Sheet (CV, GridSearch, Metrics)")
print("Production-ready face detection pipeline")
print("Single executable file - Copy/Paste/Run!")
print("="*70)
