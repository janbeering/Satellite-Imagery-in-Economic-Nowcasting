from IPython.display import Image as IPyImage
from ultralytics import YOLO
import numpy as np


###Training
def train_models(project, path, version30cm, version50cm, version50cmsnow, model_type):
    ###30cm
    version = project.version(version30cm)
    dataset = version.download(model_type)            

    IPyImage(filename=f'/{path}/train{version30cm}/confusion_matrix.png', width=600)
    IPyImage(filename=f'/{path}/train{version30cm}/results.png', width=600)
    project.version(dataset.version).deploy(model_type=model_type, model_path=f"/{path}/dist/models/train{version30cm}/")

    
    #50cm without snow
    version = project.version(version50cm)
    dataset = version.download(model_type)
                            
    IPyImage(filename=f'/insert/own/path/runs/detect/train{version50cm}/confusion_matrix.png', width=600)
    IPyImage(filename=f'/insert/own/path/runs/detect/train{version50cm}/results.png', width=600)
    project.version(dataset.version).deploy(model_type=model_type, model_path=f"/{path}/dist/models/train{version50cm}/")


    #50cm snow
    version = project.version(version50cmsnow)
    dataset = version.download(model_type)
                            
    IPyImage(filename=f'/{path}/train{version50cmsnow}/confusion_matrix.png', width=600)
    IPyImage(filename=f'/{path}/train{version50cmsnow}/results.png', width=600)
    project.version(dataset.version).deploy(model_type=model_type, model_path=f"/{path}/dist/models/train{version50cmsnow}/")


### Upload custom weights
def upload_weights(project, path, version30cm, version50cm, version50cmsnow, model_type, v_image_list):
    #20230928 -> 30cm
    version_20230928 = project.version(v_image_list[0])
    version_20230928.deploy(model_type, f"/{path}/train{version30cm}/", "weights/best.pt")

    #20230210 -> 50cm snow
    version_20230210 = project.version(v_image_list[1])
    version_20230210.deploy(model_type, f"/{path}/train{version50cmsnow}/", "weights/best.pt")

    #20220415 -> 30cm
    version_20220415 = project.version(v_image_list[2])
    version_20220415.deploy(model_type, f"/{path}/train{version30cm}/", "weights/best.pt")

    #20220325 -> 30cm
    version_20220325 = project.version(v_image_list[3])
    version_20220325.deploy(model_type, f"/{path}/train{version30cm}/", "weights/best.pt")

    #20220204 -> 50cm snow
    version_20220204 = project.version(v_image_list[4])
    version_20220204.deploy(model_type, f"/{path}/train{version50cmsnow}/", "weights/best.pt")

    #20191206 -> 50cm
    version_20191206 = project.version(v_image_list[5])
    version_20191206.deploy(model_type, f"/{path}/train{version50cm}/", "weights/best.pt")


### Validate models per image 
def validate(path, version30cm, version50cm, version50cmsnow, name_codes, v_image_list):

    dataset_path = "/insert/own/path/Satellite-Imagery-"
    """ Attention: due to the small sample size the split is on train. Change when dataset is larger."""
    
    for image in v_image_list:
        ### 20191206
        if image.isin([16, 20]):
            version = version50cm
        elif image.isin([17]):
            version = version50cmsnow
        elif image.isin([18, 19, 21]):
            version = version30cm
            
        model = YOLO(f"/{path}/train{version}/weights/best.pt")
        metric = model.val(data = f"{dataset_path}{image}/data.yaml", plots = True, split="val")
        f1_curves = metric.box.f1_curve[0:2]
        mean_cars_trucks = np.apply_along_axis(np.mean, 0, f1_curves)
        max_f1 = max(mean_cars_trucks)
        max_conf_index = mean_cars_trucks.argmax()
        max_conf = (mean_cars_trucks.argmax()+1)/1000

        #recall, precision at f1*
        precision_curves = metric.box.p_curve[0:2]
        p_value_car = precision_curves[0][max_conf_index]
        p_value_truck = precision_curves[1][max_conf_index]

        recall_curves = metric.box.r_curve[0:2]
        r_value_car = recall_curves[0][max_conf_index]
        r_value_truck = recall_curves[1][max_conf_index]
        eval_prints(version50cm, max_f1, max_conf, p_value_car, p_value_truck, r_value_car, r_value_truck)
        name_codes[0][max_conf] = max_conf

    return name_codes


def eval_prints(version50cm, max_f1, max_conf, p_value_car, p_value_truck, r_value_car, r_value_truck):
    #recall, precision at f1*
    print(f"20191206    {version50cm}")
    print("Maximal F1 value: ", max_f1)
    print("Confidence of maximal F1 value: ", max_conf)
    print("Precision at optimal confidence - Car: ", p_value_car)
    print("Precision at optimal confidence - Truck: ", p_value_truck)
    print("Recall at optimal confidence - Car: ", r_value_car)
    print("Recall at optimal confidence - Truck: ", r_value_truck)