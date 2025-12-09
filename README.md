# Sign Language Recognition System

DAEN 429 Course Project by Munish Shah

## Project Overview

Two-phase sign language recognition system implementing transfer learning with ResNet-18 for static ASL alphabet classification and temporal modeling for dynamic word recognition.

## Phase 1: Static ASL Alphabet Classification

**Dataset**: ASL Alphabet (87,000 images, 29 classes: A-Z + space/del/nothing)

**Approach**: Transfer learning with ResNet-18, progressive layer unfreezing

**Configurations Tested**:
- T-A: Head-only fine-tuning (freeze all, train fc layer)
- T-B: Last block fine-tuning (freeze stem + layer1-3, train layer4 + fc)
- T-C: Progressive fine-tuning (unfreeze layer3-4 + fc)
- S-A: Training from scratch

**Best Model**: T-C (Validation F1: 0.9997, Test Accuracy: 100%)

**Key Techniques**:
- Stratified 80/20 train/validation split (seed=429)
- BatchNorm layers kept in eval mode during frozen training
- Hyperparameter tuning for learning rate and batch size
- Model selection based on validation macro-F1 score

## Phase 2: Dynamic Word Recognition (Bonus)

**Dataset**: WLASL100 (100 word classes, video clips)

**Architecture**: ResNet-18 feature extractor + 2-layer LSTM temporal model

**Configurations Tested**:
- 2A: Freeze CNN, train temporal head only
- 2B: Unfreeze layer4, train CNN + temporal head

**Best Model**: 2B (Validation F1: 0.2342)

**Implementation**:
- 16 frames sampled per video
- Feature extraction using Phase 1 best model
- LSTM with hidden dimension 256
- Dropout regularization (0.3 in LSTM, 0.5 in classifier)

## Repository Structure

```
ASL_Classifier_Shah_Munish.ipynb  # Phase 1 implementation
Phase2_Classifier.ipynb            # Phase 2 implementation
```
