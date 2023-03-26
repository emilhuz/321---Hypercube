from detector import AnomalyDetector
from filter import ConvolutionalModel
from random import randint

class Drone:
    def __init__(self, name="drone"):
        self.name = name

    def go(self, location) -> bool:
        print(f"Flying {self.name} to {location}")
        if randint(0,1) == 1:
            print("Anomaly confirmed, alerting authorities")
            return True
        else:
            print("False alarm, returning to base")
            return False

detector = AnomalyDetector()
neural_net = ConvolutionalModel()

drones = [Drone(f"drone {i}") for i in range(10)]

def save_historical(img):
    detector.add_to_history(img)

def satellite_register_image(img):

    # find coordinates of possible areas of interest
    bounds = detector.detect_anomalies(img)

    for area in bounds:
        area_zoom = img[area[0]:area[2], area[1]:area[3]]
        # analyze in depth
        prediction = neural_net.predict(area_zoom)
        if prediction.item() == 1:
            # send drone to investigate
            drone =  drones[randint(0,len(drones)-1)]
            confirmed = drone.go(area)
            if not confirmed:
                neural_net.reveal_classification(area_zoom, 0)



