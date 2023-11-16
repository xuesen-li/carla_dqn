import cv2
import carla
import numpy as np
import time

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

actor_list = []
sensor_data = {'rgb':None}
right_start, left_start = None, None

client = carla.Client("localhost", 2000)
client.set_timeout(2.0)
world = client.load_world('Town04')
world = client.get_world()
blueprint_library = world.get_blueprint_library()
model_3 = blueprint_library.filter("model3")[0]


def process_rgb(image):
    image_4d = np.array(image.raw_data).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    sensor_data['rgb'] = image_4d
    return


try:
    vehicle = world.spawn_actor(model_3, world.get_map().get_spawn_points()[0])
    actor_list.append(vehicle)

    rgb_blueprint = blueprint_library.find('sensor.camera.rgb')
    rgb_blueprint.set_attribute("image_size_x", str(IMAGE_WIDTH))
    rgb_blueprint.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    rgb_blueprint.set_attribute("fov", f"110")
    sensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    rgb_sensor = world.spawn_actor(rgb_blueprint, sensor_transform, attach_to=vehicle)
    actor_list.append(rgb_sensor)
    rgb_sensor.listen(lambda data: process_rgb(data))

    vehicle.set_autopilot(True)

    while sensor_data['rgb'] is None:
        time.sleep(0.1)
        

    while True:
        image_rgb = cv2.cvtColor(sensor_data['rgb'], cv2.COLOR_RGBA2RGB)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) # convert to gray scale
        image_blur = cv2.GaussianBlur(image_gray, (9,9), 0) # remove noise
        image_canny = cv2.Canny(image_blur, 50, 100)
        # position filter. Only keeps the lower triangle part of the image
        polygons = np.array([[0, IMAGE_HEIGHT], 
                             [IMAGE_WIDTH, IMAGE_HEIGHT], 
                             [IMAGE_WIDTH * 0.7, IMAGE_HEIGHT * 0.5], 
                             [IMAGE_WIDTH * 0.3, IMAGE_HEIGHT * 0.5]]).astype(np.int32)
        mask = np.zeros_like(image_canny)
        cv2.fillPoly(mask, pts=[polygons], color=(255, 255, 255))
        image_canny_masked = cv2.bitwise_and(image_canny, mask)    

        # detect lines
        linesP = cv2.HoughLinesP(image_canny_masked, 1, np.pi / 180, 50, None, 50, 10)
        
        left_x, left_y, right_x, right_y = [], [], [], []
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                x1, y1, x2, y2 = l.reshape(4)
                # Fits a linear polynomial to the x and y coordinates
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                y_intercept = parameters[1]
                # seperate left and right; only keep lines with large slope(slope threshold=0.2)
                if slope > 0.2:
                    left_x.append(x1)
                    left_x.append(x2)
                    left_y.append(y1)
                    left_y.append(y2)
                    cv2.line(image_rgb, (l[0], l[1]), (l[2], l[3]), (0,255,255), 2, cv2.LINE_AA)   
                elif slope < -0.2:
                    right_x.append(x1)
                    right_x.append(x2)
                    right_y.append(y1)
                    right_y.append(y2)                    
                    cv2.line(image_rgb, (l[0], l[1]), (l[2], l[3]), (0,255, 255), 2, cv2.LINE_AA)

            if len(left_x) > 0 :
                a, b = np.polyfit(left_x, left_y, 1)
                left_start = (int((IMAGE_HEIGHT - b) / a), int(IMAGE_HEIGHT))
                left_end = (int((IMAGE_HEIGHT/1.95 - b) / a), int(IMAGE_HEIGHT/1.95))
            
            if left_start:
                cv2.line(image_rgb, left_start, left_end, (0,255,0), 1, cv2.LINE_AA)

            if len(right_x) > 0:
                a, b = np.polyfit(right_x, right_y, 1)
                right_start = (int((IMAGE_HEIGHT - b) / a), int(IMAGE_HEIGHT))
                right_end = (int((IMAGE_HEIGHT/1.95 - b) / a), int(IMAGE_HEIGHT/1.95))
            
            if right_start:
                cv2.line(image_rgb, right_start, right_end, (0,0,255), 1, cv2.LINE_AA) 

        # create output image
        tiled = np.concatenate((image_rgb, cv2.cvtColor(image_canny, cv2.COLOR_GRAY2RGB)), axis = 1) # convert gray to 3d image and make tile

        cv2.imshow("Lane Detection - Carla", tiled)
        if cv2.waitKey(10) == ord('q'):
            break
        time.sleep(0.05)

finally:
    for actor in actor_list:
        actor.destroy()