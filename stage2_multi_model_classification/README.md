# Stage 2 : Prediction Model
In Stage 2, T1-weighted MRI together with the predicted FA and MD maps from Stage 1 are used for disease classification, including binary and three-class classification tasks.

Please update file and path options in 'config.py' before execution.

## How to Run

### Training
- **Binary classification**  
python binary_train.py
- **Three-class classification**  
python 3class_train.py

### Test
- **Binary classification**  
python binary_test.py
- **Three-class classification**  
python 3class_test.py