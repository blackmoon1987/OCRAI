import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import argparse
import cv2

class OCRModel:
    def __init__(self, language: str):
        self.language = language
        self.char_list = self.get_char_list(language)
        self.model = None
        self.img_height = 64
        self.img_width = 256

    def get_char_list(self, language: str) -> list:
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
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Reshape((-1, x.shape[-1]))(x)
        
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
        
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        x = layers.multiply([x, attention])
        
        outputs = layers.Dense(len(self.char_list) + 1, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model


    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def load_image_and_label(self, image_folder: str, image_name: str):
        image_path = os.path.join(image_folder, f"{image_name}.png")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img.astype(np.float32) / 255.0
        label_path = os.path.join(image_folder, f"{image_name}.txt")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file {label_path} does not exist")
        with open(label_path, 'r', encoding='utf-8') as f:
            label = f.readline().strip()
        return img, label

    def encode_label(self, label: str):
        encoded_label = [self.char_list.index(c) for c in label if c in self.char_list]
        return np.array(encoded_label)

    def data_generator_split(self, image_folder: str, split_file: str, batch_size: int):
        with open(os.path.join(image_folder, split_file), 'r', encoding='utf-8') as f:
            image_names = [line.strip() for line in f.readlines()]

        while True:
            np.random.shuffle(image_names)
            for i in range(0, len(image_names), batch_size):
                batch_images = []
                batch_labels = []
                batch_image_names = image_names[i:i + batch_size]
                for image_name in batch_image_names:
                    try:
                        img, label = self.load_image_and_label(image_folder, image_name)
                        batch_images.append(img)
                        batch_labels.append(self.encode_label(label))
                    except Exception as e:
                        print(f"Error loading {image_name}: {e}")
                        continue
                if not batch_images:
                    continue
                label_length = np.array([len(label) for label in batch_labels])
                max_label_length = max(label_length)
                padded_labels = np.ones((len(batch_labels), max_label_length)) * -1
                for j, label in enumerate(batch_labels):
                    padded_labels[j, :len(label)] = label
                images = np.array(batch_images)
                input_length = np.ones((len(images), 1)) * (self.img_width // 4)
                label_length = label_length.reshape(-1, 1)
                yield [images, padded_labels, input_length, label_length], np.zeros(len(images))

    def train(self, image_folder: str, epochs: int = 50, batch_size: int = 32, continue_training: bool = False):
        print(f"Training model for {self.language}...")
        image_names = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(".png")]
        np.random.shuffle(image_names)
        split_index = int(0.8 * len(image_names))
        train_image_names = image_names[:split_index]
        val_image_names = image_names[split_index:]

        def save_split(image_names, split_type):
            split_file = os.path.join(image_folder, f'{split_type}_split.txt')
            with open(split_file, 'w', encoding='utf-8') as f:
                for name in image_names:
                    f.write(f"{name}\n")

        save_split(train_image_names, 'train')
        save_split(val_image_names, 'val')

        train_generator = self.data_generator_split(image_folder, 'train_split.txt', batch_size)
        val_generator = self.data_generator_split(image_folder, 'val_split.txt', batch_size)

        steps_per_epoch = len(train_image_names) // batch_size
        validation_steps = len(val_image_names) // batch_size

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
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            epochs=epochs
        )

        self.model.save(f'ocr_model_{self.language}.h5')
        print(f"Model for {self.language} trained and saved.")
        return history

    def load_model_file(self):
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
            result = tf.strings.reduce_join([self.char_list[int(c)] for c in result if c != -1]).numpy().decode('utf-8')
            output_text.append(result)
        return output_text

    def extract_text(self, image_path: str) -> str:
        if self.model is None:
            self.load_model_file()
        if self.model is None:
            return f"No model available for {self.language}"
        preprocessed_image = self.preprocess_image(image_path)
        prediction = self.model.predict(preprocessed_image)
        decoded_text = self.decode_predictions(prediction)[0]
        return decoded_text

class MultilingualOCRSystem:
    def __init__(self):
        self.models = {}

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
