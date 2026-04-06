# HandSign — Sign Language Digit Recognizer
### Sakshi Saraswat | GWECA, Ajmer

A deep learning system that recognizes hand signs for digits 0–9, built to help deaf and mute learners practice sign language through AI-powered feedback.

---

## Real-World Problem
Over 63 million people in India have significant hearing disability. Learning sign language digits is a fundamental step for communication, but practice tools are scarce. HandSign uses a neural network to instantly recognize hand sign images and tell the learner which digit they are showing.

## Dataset
**Sign Language MNIST** — available on Kaggle  
`kaggle datasets download datamunge/sign-language-mnist`

- 27,455 training samples, 7,172 test samples
- 28×28 grayscale images of hand signs
- 10 classes (digits 0–9)

## Model
Fully connected neural network with:
- 4 hidden layers (512 → 256 → 128 → 64 neurons)
- Batch Normalization + Dropout for regularization
- Adam optimizer, categorical crossentropy loss
- EarlyStopping + ReduceLROnPlateau callbacks

## Results
| Metric | Value |
|---|---|
| Test Accuracy | ~93–95% |
| Loss | ~0.18 |

## How to Run
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

# Download dataset from Kaggle and place CSVs in same folder
python handsign.py
```

## Output Files
- `sample_images.png` — sample sign images per class
- `training_curves.png` — accuracy and loss over epochs
- `confusion_matrix.png` — per-class prediction accuracy
- `predictions.png` — sample predictions with confidence
- `handsign_model.h5` — saved trained model

## Tech Stack
Python · TensorFlow/Keras · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn
