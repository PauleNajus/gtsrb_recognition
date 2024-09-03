import os

class Config:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{BASE_PATH}/traffic_signs.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_PATH, 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'ppm'}

    CNN_EPOCHS = 20
    CNN_BATCH_SIZE = 32
    CNN_LEARNING_RATE = 0.001

    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10

    IMAGE_SIZE = (32, 32)
    AUGMENTATION_FACTOR = 2

    TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, 'data', 'train', 'GTSRB_Final_Training_Images')
    TEST_IMAGES_PATH = os.path.join(BASE_PATH, 'data', 'test', 'GTSRB_Final_Test_Images')
    TEST_GT_PATH = os.path.join(BASE_PATH, 'data', 'test', 'GTSRB_Final_Test_GT', 'GT-final_test.csv')

    CLASS_NAMES = [
        'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
        'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
        'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]

    @classmethod
    def get_train_folders(cls):
        return [f'{i:05d}' for i in range(43)]

    @classmethod
    def get_train_csv_paths(cls):
        return [os.path.join(cls.TRAIN_IMAGES_PATH, f'GT-{i:05d}.csv') for i in range(43)]

    @classmethod
    def get_train_image_paths(cls):
        return [os.path.join(cls.TRAIN_IMAGES_PATH, file)
                for file in os.listdir(cls.TRAIN_IMAGES_PATH)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.ppm'))]

    @classmethod
    def get_test_image_paths(cls):
        return [os.path.join(cls.TEST_IMAGES_PATH, f) for f in os.listdir(cls.TEST_IMAGES_PATH) if f.endswith('.ppm')]