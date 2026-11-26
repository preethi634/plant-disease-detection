# AI Plant Doctor – Dataset Audit

## 1. Project Objective
To collect, evaluate and organize freely available agricultural image datasets for Indian crop disease detection.

## 2. Existing Dataset
- Source: PlantVillage
- Crops: Tomato, Potato and Bell Pepper
- Images: 20,638
- Classes: 15
- Existing Model: MobileNetV2
- Test Accuracy: 93.77%

## 3. Additional Dataset Sources

| Dataset | Target Crops | Status |
|---|---|---|
| Digital Green | Chilli, Brinjal, Mango, Maize, Papaya, Wheat and others | To audit |
| Multi-Crop Disease Dataset | Banana, Chilli, Groundnut, Radish, Cauliflower | To audit |
| ICAR Dataset | Rice and Maize | To audit |

## 4. Dataset Evaluation Criteria
- Image quantity and quality
- Crop and disease labels
- Healthy and diseased image distribution
- Duplicate images
- Image resolution
- Dataset licensing and attribution
- Suitability for real-world Indian agricultural conditions

## 5. Dataset Collection Status
Initial research completed. Dataset inspection and validation pending.


## 6. Digital Green Dataset – Initial Inspection

- Dataset source: Hugging Face – DigiGreen/Crop_Disease_Images
- Records: Approximately 1,026
- Crop categories: 74
- Diagnosis categories: 145
- Country: India
- Image format: JPEG
- License: CC BY 4.0

### Sample Image Inspection

- Image: 00387.jpg
- Crop: Chilli
- Diagnosis: Thrips
- State: Andhra Pradesh
- Observed symptoms: Upward curling and deformation of leaves, stunted growth.
- Initial assessment: Symptoms are consistent with the written description, but are not sufficient to independently confirm thrips.

### Quality Issues Identified

1. Some diagnosis labels conflict with their descriptions.
2. Some records contain multiple diagnoses.
3. Some descriptions are incomplete or ambiguous.
4. Some descriptions contain incorrect or insufficient information.
5. Some records may require expert verification.

### Preliminary Decision

Conditionally suitable for carefully reviewed training and real-world evaluation. Do not use all records without label validation.
