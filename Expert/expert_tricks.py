import cv2
import glob
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv
from skimage.measure import regionprops

# T1 start ________________________________________________________________
# Read Dataset

# Change the dataset path here according to your folder structure
dataset_path = "C:\\Users\\18340\\Desktop\\Mine\\Study\\Projects\\NUS_Summer_camp\\VisualComputing\\TrafficSignRecognition\\Dataset_1\\images\\"

X = []
y = []
for i in glob.glob(dataset_path + '*.png', recursive=True):
    label = i.split("images")[1][1:4]
    y.append(label)

    # Read each file 'i' and append it to the list 'X'
    img = cv2.imread(i)
    X.append(img)

# You should have X and y with 5998 entries each.
# T1 end _________________________________________________________________


# T2 start ________________________________________________________________
# Preprocessing
X_processed = []
for x in X:
    # Resize image 'x' to 48x48 and store it in 'temp_x'
    temp_x = cv2.resize(x, (48, 48))
    # Convert 'temp_x' to grayscale
    temp_x = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)
    # Apply additional preprocessing techniques (e.g., Gaussian blur, histogram equalization) here if desired
    temp_x = cv2.GaussianBlur(temp_x, (3, 3), 0)
    temp_x = cv2.equalizeHist(temp_x)
    # Normalize the pixel values of the image
    temp_x = temp_x.astype(np.float32) / 255.0
    # Append the preprocessed image to 'X_processed'
    X_processed.append(temp_x)

# T2 end _________________________________________________________________


# T3 start ________________________________________________________________
# Feature extraction
X_features = []
for x, x_processed in zip(X, X_processed):
    # Apply different feature extraction methods here
    # Example: Histogram of Oriented Gradients (HOG)
    hog_feature = hog(x_processed, orientations=8, pixels_per_cell=(10, 10),
                      cells_per_block=(1, 1), visualize=False, multichannel=False)

    # Color distribution features (HSV color space)
    hsv_image = rgb2hsv(x)  # Convert image to HSV color space
    hue_hist = np.histogram(hsv_image[:, :, 0], bins=8, range=(0, 1))[0]  # Hue histogram
    saturation_hist = np.histogram(hsv_image[:, :, 1], bins=8, range=(0, 1))[0]  # Saturation histogram
    value_hist = np.histogram(hsv_image[:, :, 2], bins=8, range=(0, 1))[0]  # Value histogram

    # Shape features
    label_img = np.uint8(x_processed > 0)  # Convert image to binary label image
    props = regionprops(label_img)[0]  # Calculate region properties
    area = props.area  # Area of the region
    perimeter = props.perimeter  # Perimeter of the region
    eccentricity = props.eccentricity  # Eccentricity of the region

    # Concatenate all features into a single feature vector
    x_features = np.concatenate((hog_feature, hue_hist, saturation_hist, value_hist, [area, perimeter, eccentricity]))

    X_features.append(x_features)


# Apply dimensionality reduction using PCA
pca = PCA(n_components=100)  # Adjust the number of components as needed
X_features = pca.fit_transform(X_features)


# Split training and testing sets using sklearn.model_selection.train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# T3 end _________________________________________________________________


# T4 start ________________________________________________________________
# Train and evaluate different models
models = [
    SVC(),  # Support Vector Classifier (SVC)
    RandomForestClassifier(),  # Random Forest Classifier
    KNeighborsClassifier(),  # k-Nearest Neighbors Classifier
    DecisionTreeClassifier(),  # Decision Tree Classifier
    GaussianNB(),  # Naive Bayes Classifier
    MLPClassifier(max_iter=500)  # Artificial Neural Network (ANN)
]

for model in models:
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model: {model.__class__.__name__}, Accuracy: {accuracy}")

# T4 end _________________________________________________________________
