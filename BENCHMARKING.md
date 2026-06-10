# OncoResolve: Clinical & Technical Benchmarking Report

This report benchmarks the `OncoResolve` framework against established molecular subtyping methods, gene expression signatures, and open-source R packages.

---

## 1. Executive Summary

`OncoResolve` bridges a critical gap in the bioinformatics ecosystem by providing a **high-hygiene, Python-native** library for breast cancer subtyping and consensus prognosis. 

Historically, clinical researchers had to rely on proprietary commercial assays (e.g. Oncotype DX, MammaPrint) or R-based packages (e.g. `genefu`, `AIMS`) that are difficult to integrate into modern Python-based machine learning and data science pipelines. `OncoResolve` introduces pre-trained, cohort-independent machine learning classifiers (RBF-SVM and Logistic Regression), a consensus prognostic model (Ridge Cox), and a patient uniqueness scorer (CUS).

---

## 2. Molecular Subtyping Methods Benchmarking

Here we compare `OncoResolve`'s subtyping module (`OncoClassifier`) against established subtyping methods:

| Method / Algorithm | Algorithm Class | Gene Count | Platform Invariance | Cohort Dependence | Single Sample (n-of-1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PAM50 (Standard)** | Nearest Centroid (Pearson/Spearman) | 50 genes | Low | High (requires median-centering against a balanced cohort) | No (fails on single samples without reference) |
| **PCAPAM50** | Principal Component Alignment + Centroid | 50 genes | Medium | High | No |
| **AIMS** | Rule-based (Absolute Intrinsic) | 50 gene-pairs | High | Low (uses simple pairwise binary rules) | Yes |
| **SCMGENE / SSP** | Nearest Centroid / Single Sample Predictor | 50 genes | Medium | Medium | Yes (if pre-standardized) |
| **IntClust** | Joint clustering (Copy number + Expression) | Genomic scale | Low | High | No |
| **Claudin-low** | Correlation centroid | 9 cell-type genes | Low | High | No |
| **CIT Classification** | Distance-based centroid | 344 genes | Medium | High | No |
| **BluePrint** | Nearest Centroid | 80 genes | Low | High | No |
| **OncoResolve** | **Machine Learning (RBF-SVM / Logistic Regression)** | **178 genes** | **High** (pipeline includes a pre-trained TCGA StandardScaler) | **None** (zero cohort dependence for pre-trained classifiers) | **Yes** (robustly subtypes a single patient sample) |

### Key Takeaway on Subtyping:
* **The Cohort Dependency Problem:** Standard PAM50 and centroid methods (like `genefu`) calculate the correlation of a sample's gene expression against subtype centroids. However, correlation is highly sensitive to cohort composition. If you run PAM50 on a cohort that contains 90% Luminal A patients, the median-centering step will shift the baseline and misclassify Luminal A patients as Basal or HER2.
* **OncoResolve's Solution:** By training an **RBF-SVM** and **Logistic Regression** model on the TCGA-BRCA cohort and embedding the fitted `StandardScaler` inside the pipeline, `OncoResolve` classifies each sample based on its absolute placement in the high-dimensional boundary space. This makes it a true **Single-Sample Predictor (SSP)** that can subtype an individual patient in a clinical clinic setting without requiring a reference cohort.

---

## 3. Prognostic Signatures Benchmarking

Here we compare `OncoResolve`'s survival module (`OncoPrognosis`) against established prognostic signatures:

| Signature / Test | Clinical Utility | Gene Count | Output Format | Software Availability | Proprietary / Open-Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Oncotype DX** | Chemo benefit / 10-yr recurrence risk | 21 genes | Recurrence Score (RS, 0-100) | None | Proprietary ($4,000+ per test) |
| **MammaPrint** | 10-year distant metastasis risk | 70 genes | Binary High vs. Low Risk | None | Proprietary (Commercial) |
| **EndoPredict** | 10-year recurrence risk (ER+/HER2-) | 12 genes | EndoPredict Score (EPclin) | None | Proprietary (Commercial) |
| **Prosigna (PAM50)** | Subtyping + Recurrence risk | 50 genes | Risk of Recurrence (ROR-S/P/T) | None | Proprietary (FDA-cleared) |
| **Breast Cancer Index (BCI)** | Late recurrence risk (5-10 years) | 7 genes | BCI Prognostic Score | None | Proprietary (Commercial) |
| **Rotterdam 76** | Node-negative metastasis risk | 76 genes | High vs. Low Risk classification | R (`genefu`) | Open-Source |
| **OncoResolve** | **Subtyping + Overall Survival Prediction** | **178 genes** | **Consensus Risk Score (CRS)** | **Python (`OncoResolve`)** | **Open-Source (Free)** |

### Key Takeaway on Prognosis:
* Commercial tests (Oncotype DX, MammaPrint, Prosigna) are extremely expensive and performed in centralized laboratories, limiting their use in academic research and low-resource clinical trials.
* `OncoResolve` replicates the prognostic capabilities of these assays by bundling a regularized **Ridge Cox Proportional Hazards model** trained on TCGA-BRCA survival outcomes. It outputs a **Consensus Risk Score (CRS)** that allows researchers to risk-stratify cohorts using standard bulk RNA-seq data without any licensing costs.

---

## 4. Software & Package Ecosystem Benchmarking

Here we compare the software architecture of `OncoResolve` with open-source R implementations:

| R Package / Tool | Primary Capabilities | Base Language | Machine Learning Options | Batch Correction | Unified Workflow (Subtyping + Prognosis) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **genefu** | PAM50, AIMS, Rotterdam76 signatures | R | None (correlation-based only) | No | Yes |
| **pamr** | Nearest shrunken centroids | R | Shrunken Centroids | No | No (general classifier) |
| **AIMS** | Rule-based subtyping | R | Binary rules | Yes (by nature of pairwise rules) | No |
| **ConsensusClusterPlus** | Consensus clustering for discovery | R | Unsupervised clustering | No | No (discovery tool) |
| **NMF / SNFtool** | Non-negative Matrix / Similarity Network | R / Matlab | Unsupervised clustering | No | No (discovery tool) |
| **CancerSubtypes** | Clustering wrapper for subtyping discovery | R | Silhouette, NMF, CNMF | No | No (discovery tool) |
| **MOVICS** | Multi-omics subtyping discovery | R | NMF, iCluster, SNF, Consensus | No | No (discovery tool) |
| **OncoResolve** | **PAM50 subtyping, Prognosis, Outliers** | **Python** | **RBF-SVM, Logistic Regression, Ridge Cox** | **Yes** (`scale_cohort` + internal scaler option) | **Yes** (unified API for Subtyping, CRS, and CUS) |

### Key Takeaway on Software:
* **The R-to-Python Gap:** Modern bioinformatics, machine learning, and deep learning libraries (TensorFlow, PyTorch, Scikit-Learn) are primarily written in Python. R packages like `genefu` require R-to-Python bridges (like `rpy2`), which are notoriously difficult to configure, slow, and prone to installation crashes.
* **OncoResolve's Solution:** Built natively in Python on top of `scikit-learn` and `lifelines`, `OncoResolve` integrates seamlessly into modern machine learning workflows (e.g. feeding subtyping probabilities as features into downstream neural networks) with zero setup friction.
