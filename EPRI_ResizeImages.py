import cv2
import os


# Try to get directory path
try:
    home_path = os.getcwd()
    # print(home_path)
except:
    # This will be changed later to ask for the path
    print("Cannot find home path, no path finder implemented yet.\n"
          "Make sure that you use 'import os' in the Python file")

SourceDic = {
    "Broken" : home_path + "/Training_SolarImages/Data/Broken",
    "Cracked" : home_path + "/Training_SolarImages/Data/Cracked/",
    "Good" : home_path + "/Training_SolarImages/Data/Good/",
    "Hot" : home_path + "/Training_SolarImages/Data/Hot/",
    "Dirty": home_path + "/Training_SolarImages/Data/Dirty/"
}

PredicDic = {
    "Broken" : home_path + "/Training_SolarImages/Predict/Broken/",
    "Cracked" : home_path + "/Training_SolarImages/Predict/Cracked/",
    "Good" : home_path + "/Training_SolarImages/Predict/Good/",
    "Hot" : home_path + "/Training_SolarImages/Predict/Hot/",
    "Dirty": home_path + "/Training_SolarImages/Predict/Dirty/"
}


# source_dir = SourceDic["Good"]
source_dir = "C:/Users/Adam/PycharmProjects/EPRIProjectV2/Training_SolarImages/Data/RealTest/Good"
target_dir = "C:/Users/Adam/PycharmProjects/EPRIProjectV2/Training_SolarImages/RealTest/Good"
target_size = (512, 512)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for filename in os.listdir(source_dir):
    filepath = os.path.join(source_dir, filename)
    img = cv2.imread(filepath)
    img = cv2.resize(img, target_size)
    target_filepath = os.path.join(target_dir, filename)
    cv2.imwrite(target_filepath, img)