# OCRAI (Optical Character Recognition AI)

OCRAI is a multilingual OCR system using Python and deep learning. It supports training and prediction for various languages, including English and Arabic.

## Setup
1. Clone: git clone https://github.com/blackmoon1987/OCRAI.git
2. Install: pip install -r requirements.txt (numpy, tensorflow, opencv-python, argparse)

## Data Organization
- Create language folders (e.g., enToLearn, arToLearn)
- Place matching .png images and .txt files in these folders

## Usage
Training: python main.py --mode train --language [lang_code] --image_folder [folder_path] --epochs [num] --batch_size [num]
Example: python main.py --mode train --language en --image_folder ./enToLearn --epochs 50 --batch_size 32
Continue training: Add --continue_training

Prediction: python main.py --mode predict --language [lang_code] --image_path [image_path]
Example: python main.py --mode predict --language en --image_path ./enToLearn/image1.png

## Project Structure
OCRAI/
├── main.py
├── requirements.txt
├── enToLearn/ (English training data)
├── arToLearn/ (Arabic training data)

## Tips
- Ensure image-text pair names match
- Use clear images for better accuracy
- Diversify training data
- Monitor training to avoid overfitting
- Save models after training

## Troubleshooting
- "0 images found": Check file existence and permissions
- Model loading errors: Verify correct saving and character list
- Text formatting: Ensure correct content in text files

For help, open an issue on GitHub.
