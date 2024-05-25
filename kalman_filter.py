import cv2
import carla
import numpy as np
import time
import matplotlib.pyplot as plt
import random

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


def rgb_callback(image):
    image_4d = np.array(image.raw_data).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    sensor_data['rgb'] = image_4d
    return

def gnss_callback(data):
    sensor_data['gnss'] = {
        'timestamp': data.timestamp,
        'location': data.transform.location
    }

def imu_callback(data):
    sensor_data['imu'] = {
        'timestamp': data.timestamp,
        'gyro': data.gyroscope,
        'accel': data.accelerometer,
        'compass': data.compass
    }

try:
    spawn_point = random.choice(world.get_map().get_spawn_points())
    print(spawn_point.location)
    vehicle = world.spawn_actor(model_3, spawn_point)
    actor_list.append(vehicle)

    rgb_blueprint = blueprint_library.find('sensor.camera.rgb')
    rgb_blueprint.set_attribute("image_size_x", str(IMAGE_WIDTH))
    rgb_blueprint.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    rgb_blueprint.set_attribute("fov", f"110")
    sensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    rgb_sensor = world.spawn_actor(rgb_blueprint, sensor_transform, attach_to=vehicle)
    actor_list.append(rgb_sensor)
    rgb_sensor.listen(lambda data: rgb_callback(data))

    # Add navigation sensor
    gnss_blueprint = blueprint_library.find('sensor.other.gnss')
    gnss_blueprint.set_attribute("sensor_tick", "0.5")
    gnss_sensor = world.spawn_actor(gnss_blueprint, carla.Transform(), attach_to=vehicle)
    actor_list.append(gnss_sensor)
    gnss_sensor.listen(lambda data: gnss_callback(data))

    # Add inertial measurement sensor
    imu_blueprint = blueprint_library.find('sensor.other.imu')
    imu_sensor = world.spawn_actor(imu_blueprint, carla.Transform(), attach_to=vehicle)
    actor_list.append(imu_sensor)
    imu_sensor.listen(lambda data: imu_callback(data))    

    vehicle.set_autopilot(True)

    while sensor_data['rgb'] is None:
        time.sleep(0.1)
        
    fig, ax = plt.subplots(2, 2)
    
    step = 0


    # kalman parameters
    dt = 0.115
    F = np.array([[1, 0, dt, 0, 0.5*dt*dt, 0],
                 [0, 1, 0, dt, 0, 0.5*dt*dt],
                 [0, 0, 1, 0, dt, 0],
                 [0, 0, 0, 1, 0, dt],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]])
    
    H = np.array([[1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]])

    Q = np.array([[1, 0.5, 0, 0, 0, 0],
                  [0.5, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0.5, 0, 0],
                  [0, 0, 0.5, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0.5],
                  [0, 0, 0, 0, 0.5, 1]]) / 10.0

    R = np.array([[1, 0.5, 0, 0],
                  [0.5, 1, 0, 0],
                  [0, 0, 5, 2.5],
                  [0, 0, 2.5, 5]]) /10.0

    x_prev = np.array([[sensor_data['gnss']['location'].x,
                       sensor_data['gnss']['location'].y,
                       0,
                       0,
                       sensor_data['imu']['accel'].x,
                       sensor_data['imu']['accel'].y]]).T
    
    P_prev = np.diag([1,1,1,1,1,1])

    while True:
        step += 1
        
        # kalman filter: predict
        x_predict = F @ x_prev
        P_predict = F @ P_prev @ F.T + Q

        # kalman filter: update
        z = np.array([[sensor_data['gnss']['location'].x,
                      sensor_data['gnss']['location'].y,
                      sensor_data['imu']['accel'].x,
                      sensor_data['imu']['accel'].y]]).T
        K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + R)
        x_update = x_predict + K @ (z - H @ x_predict)
        P_update = P_predict - K @ H @ P_predict

        # kalman filter: prepare variables for next iter
        x_prev = x_update
        P_prev = P_update

        # print(x_prev)
        
        image_rgb = cv2.cvtColor(sensor_data['rgb'], cv2.COLOR_RGBA2RGB)
        cv2.imshow("Lane Detection - Carla", image_rgb)
        if cv2.waitKey(10) == ord('q'):
            break

        ax[0,0].plot(sensor_data['imu']['timestamp'], sensor_data['imu']['accel'].x, 'o--', color='green', markersize=1)
        ax[0,0].plot(sensor_data['imu']['timestamp'], sensor_data['imu']['accel'].y, 'o--', color='blue', markersize=1)
        ax[0,0].legend(['imu-x', 'imu-y'], loc="upper right")
        
        ax[0,1].plot(sensor_data['gnss']['location'].x, sensor_data['gnss']['location'].y, 'o--', color='blue', markersize=1)
        ax[0,1].plot(x_update[0], x_update[1], 'o--', color='red', markersize=1)
        ax[0,1].legend(['gnss-location', 'kf-location'], loc="upper right")

        v = vehicle.get_velocity()
        ax[1,0].plot(sensor_data['imu']['timestamp'], v.x, 'o--', color='blue', markersize=1)
        ax[1,0].plot(sensor_data['imu']['timestamp'], x_update[2], 'o--', color='red', markersize=1)  
        ax[1,0].legend(['vel-x', 'kf-vel-x'], loc="upper right")      

        ax[1,1].plot(sensor_data['imu']['timestamp'], v.y, 'o--', color='blue', markersize=1)
        ax[1,1].plot(sensor_data['imu']['timestamp'], x_update[3], 'o--', color='red', markersize=1) 
        ax[1,1].legend(['vel-y', 'kf-vel-y'], loc="upper right") 

        plt.show(block=False)
        plt.pause(0.1)


finally:
    for actor in actor_list:
        actor.destroy()