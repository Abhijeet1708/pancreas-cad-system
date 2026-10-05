# Pancreatic Cancer CAD Prototype - Scope

## In Scope
* Development of an end-to-end 4-stage research pipeline (Preprocessing, Segmentation, Classification, Explainability) for CT scans.
* Training and evaluation on publicly accessible datasets (e.g., Medical Segmentation Decathlon Task07_Pancreas or similar).
* Basic documentation and setup instructions for reproducing the pipeline.

## Out of Scope
* Multi-modal imaging (e.g., PET, MRI) unless natively provided in the chosen dataset.
* Production-level optimizations, real-time inference, or UI/UX development.
* Integration with clinical PACS or Electronic Health Record (EHR) systems.

## Constraints
* **Compute Restrictions:** The system architecture and training processes must be lightweight enough to run entirely on free-tier GPU resources (e.g., Google Colab, Kaggle).
* **Timeline:** All development, training, and documentation must be completed within a 2-month timeline.

## Explicit Limitations
* **RESEARCH PROTOTYPE ONLY:** This software is strictly an experimental prototype for research purposes.
* **NOT CLINICALLY VALIDATED:** The system has not been validated in a clinical setting and must not be used for diagnostic decision-making, patient care, or any clinical applications.
