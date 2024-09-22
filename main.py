import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Dict, List
import argparse
import cv2

class OCRModel:
    def __init__(self, language: str):
        self.language = language
        self.char_list = self.get_char_list(language)
        self.model = None
        self.img_height = 64
        self.img_width = 256

    def get_char_list(self, language: str) -> List[str]:
        language = language.lower()
        char_lists = {
            'en': list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
            'ar': list('أبتثجحخدذرزسشصضطظعغفقكلمنهوي٠١٢٣٤٥٦٧٨٩'),
            'zh': list('的一是不了人我在有他这为之大来以个中上们'),
            'hi': list('अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह०१२३४५६७८९'),
            'es': list('abcdefghijklmnñopqrstuvwxyzáéíóúüABCDEFGHIJKLMNÑOPQRSTUVWXYZÁÉÍÓÚÜ0123456789'),
            'fr': list('abcdefghijklmnopqrstuvwxyzàâäæçéèêëîïôœùûüÿABCDEFGHIJKLMNOPQRSTUVWXYZÀÂÄÆÇÉÈÊËÎÏÔŒÙÛÜŸ0123456789'),
            'de': list('abcdefghijklmnopqrstuvwxyzäöüßABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789'),
            'ja': list('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'),
            'ru': list('абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789'),
            'pt': list('abcdefghijklmnopqrstuvwxyzáâãàçéêíóôõúABCDEFGHIJKLMNOPQRSTUVWXYZÁÂÃÀÇÉÊÍÓÔÕÚ0123456789'),
            'it': list('abcdefghijklmnopqrstuvwxyzàèéìíîòóùúABCDEFGHIJKLMNOPQRSTUVWXYZÀÈÉÌÍÎÒÓÙÚ0123456789'),
            'ko': list('ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ'),
            'tr': list('abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ0123456789'),
            'nl': list('abcdefghijklmnopqrstuvwxyzäëïöüABCDEFGHIJKLMNOPQRSTUVWXYZÄËÏÖÜ0123456789'),
            'sv': list('abcdefghijklmnopqrstuvwxyzåäöABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ0123456789'),
            'pl': list('aąbcćdeęfghijklłmnńoóprsśtuwyzźżAĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ0123456789'),
            'vi': list('aăâbcdđeêghiklmnoôơpqrstuưvxyAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXY0123456789'),
            'th': list('กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'),
        }
        if language in char_lists:
            return char_lists[language]
        else:
            supported_languages = ', '.join(char_lists.keys())
            raise ValueError(f"Unsupported language: {language}. Supported languages are: {supported_languages}")

    def create_crnn_model(self):
        inputs = layers.Input(shape=(self.img_height, self.img_width, 3))
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Reshape((-1, x.shape[-1]))(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        outputs = layers.Dense(len(self.char_list) + 1, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def train(self, image_folder: str, epochs: int = 50, batch_size: int = 32, continue_training: bool = False):
        print(f"Training model for {self.language}...")
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=0.2
        )
        train_generator = train_datagen.flow_from_directory(
            image_folder,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='sparse',
            subset='training'
        )
        validation_generator = train_datagen.flow_from_directory(
            image_folder,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation'
        )
        if continue_training and self.model is not None:
            print("Continuing training from the previous model...")
        else:
            self.model = self.create_crnn_model()
        labels = layers.Input(name='labels', shape=[None], dtype='float32')
        input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
        label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
        loss_out = layers.Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [self.model.output, labels, input_length, label_length])
        model = models.Model(inputs=[self.model.input, labels, input_length, label_length], outputs=loss_out)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
        if continue_training:
            model_path = f'ocr_model_{self.language}.h5'
            if os.path.exists(model_path):
                self.model.load_weights(model_path)
                print(f"Loaded weights from previous training for {self.language}")
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=epochs
        )
        self.model.save(f'ocr_model_{self.language}.h5')
        print(f"Model for {self.language} trained and saved.")
        return history

    def load_model(self):
        model_path = f'ocr_model_{self.language}.h5'
        if os.path.exists(model_path):
            self.model = models.load_model(model_path, custom_objects={'ctc_lambda_func': self.ctc_lambda_func})
            print(f"Model for {self.language} loaded.")
        else:
            print(f"No saved model found for {self.language}.")

    def preprocess_image(self, image_path: str):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def decode_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(self.char_list[result]).numpy().decode('utf-8')
            output_text.append(result)
        return output_text

    def extract_text(self, image_path: str) -> str:
        if self.model is None:
            self.load_model()
        if self.model is None:
            return f"No model available for {self.language}"
        preprocessed_image = self.preprocess_image(image_path)
        prediction = self.model.predict(preprocessed_image)
        decoded_text = self.decode_predictions(prediction)[0]
        return decoded_text

class MultilingualOCRSystem:
    def __init__(self):
        self.models: Dict[str, OCRModel] = {}

    def add_model(self, language: str):
        if language not in self.models:
            self.models[language] = OCRModel(language)

    def train_model(self, language: str, image_folder: str, epochs: int = 50, batch_size: int = 32, continue_training: bool = False):
        if language not in self.models:
            self.add_model(language)
        return self.models[language].train(image_folder, epochs, batch_size, continue_training)

    def extract_text(self, language: str, image_path: str) -> str:
        if language not in self.models:
            self.add_model(language)
        return self.models[language].extract_text(image_path)

def main():
    parser = argparse.ArgumentParser(description="Multilingual OCR Training and Prediction")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g., 'en', 'ar', 'zh')")
    parser.add_argument("--mode", type=str, choices=['train', 'predict'], required=True, help="Mode: 'train' or 'predict'")
    parser.add_argument("--image_folder", type=str, help="Folder containing training images")
    parser.add_argument("--image_path", type=str, help="Path to the image for prediction")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--continue_training", action='store_true', help="Continue training from a previous model")
    args = parser.parse_args()

    system = MultilingualOCRSystem()

    if args.mode == 'train':
        if not args.image_folder:
            raise ValueError("Image folder is required for training mode")
        system.train_model(args.language, args.image_folder, args.epochs, args.batch_size, args.continue_training)
    elif args.mode == 'predict':
        if not args.image_path:
            raise ValueError("Image path is required for prediction mode")
        extracted_text = system.extract_text(args.language, args.image_path)
        print(f"Extracted Text: {extracted_text}")

if __name__ == "__main__":
    main()
