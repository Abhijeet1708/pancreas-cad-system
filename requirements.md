# Pancreatic Cancer CAD Prototype - Requirements

## Functional Requirements
* **Stage 1: Preprocessing:** The system must load medical imaging volumes (e.g., NIfTI), apply Hounsfield Unit (HU) windowing tailored for the pancreas/abdominal soft tissue, and perform intensity normalization and spatial resampling.
* **Stage 2: Segmentation:** The system must segment the pancreas and potential lesions from the preprocessed volumes.
* **Stage 3: Classification:** The system must classify the segmented regions or the overall volume to predict the probability or presence of pancreatic cancer.
* **Stage 4: Explainability:** The system must generate visual explanations (e.g., Grad-CAM, saliency maps) to highlight the regions that most influenced the classification outcome.

## Non-Functional Requirements
* **Modularity:** The codebase must be highly modular, allowing independent testing and execution of the four distinct stages.
* **Technology Stack:** Must be Python-based, relying on open-source libraries (e.g., PyTorch, MONAI, SimpleITK).

## Performance Targets
* **Segmentation:** Dice Similarity Coefficient (DSC) > 0.6.
* **Classification:**
  * Area Under the ROC Curve (AUC) > 0.75
  * Accuracy (Target > 0.70)
  * Sensitivity/Recall (Target > 0.70)
  * Specificity (Target > 0.70)
  * F1-score (Target > 0.70)
