import os
import random
import time
import numpy as np
import cv2
import math
import datetime
from collections import deque

import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, AveragePooling2D, Flatten, Activation
from keras.applications.xception import Xception
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard

from threading import Thread
from tqdm import tqdm
import carla

############# constants ############
EPISODES = 100
EPSILON_DECAY = 0.95 ## 0.95 0.9975 99975
MIN_EPSILON = 0.001

NUMBER_OF_ACTIONS = 3

IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200.0

DISCOUNT = 0.99
epsilon = 1

AGGREGATE_STATS_EVERY = 10

SHOW_PREVIEW = True
SHOW_PREVIEW_EVERY = 10

############# class definition ############
class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    rgb_sensor_data = None
    sem_sensor_data = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.vehicle = self.world.spawn_actor(self.model_3, \
                                              random.choice(self.world.get_map().get_spawn_points()))
        self.actor_list.append(self.vehicle)

        self.rgb_blueprint = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_blueprint.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_blueprint.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_blueprint.set_attribute("fov", f"110")
        sensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.rgb_sensor = self.world.spawn_actor(self.rgb_blueprint, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_sensor)
        self.rgb_sensor.listen(lambda data: self.process_rgb(data))

        self.sem_blueprint = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_blueprint.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_blueprint.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_blueprint.set_attribute("fov", f"110")
        sensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sem_sensor = self.world.spawn_actor(self.sem_blueprint, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.sem_sensor)
        self.sem_sensor.listen(lambda data: self.process_sem(data))                
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(1)

        col_blueprint = self.blueprint_library.find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(col_blueprint, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda data: self.process_col(data))

        while self.rgb_sensor_data is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.rgb_sensor_data

    def process_col(self, data):
        self.collision_hist.append(data)

    def process_rgb(self, image):
        image_4d = np.array(image.raw_data).reshape((self.im_height, self.im_width, 4))
        self.rgb_sensor_data = image_4d[:, :, :3]

    def process_sem(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        image_4d = np.array(image.raw_data).reshape((self.im_height, self.im_width, 4))
        self.sem_sensor_data = image_4d[:, :, :3]     

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.5*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.5*self.STEER_AMT))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-0.5*self.STEER_AMT))   
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0))
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.5*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        w = self.vehicle.get_angular_velocity() # angular velocity in deg/s
        # print("    angular vel = ", abs(w.z))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1.0
        else:
            done = False
            reward = 1.0
            # if abs(w.z) < 50:
            #     reward = 1.0 / 200.0
            # else:
            #     reward = 0.5 / 200.0

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.rgb_sensor_data, reward, done, None


class DQNAgent:
    def __init__(self):
        self.model = self.create_model("Xception")
        self.target_model = self.create_model("Xception")
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0
       
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def create_model(self, model_type):
        if model_type == "Xception":
            base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH,3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(NUMBER_OF_ACTIONS, activation="linear")(x)
            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        
        # elif model_type == "64x3_CNN":
        #     base_model = Sequential()

        #     base_model.add(Conv2D(64, (3, 3), input_shape=(IM_HEIGHT, IM_WIDTH,3), padding='same'))
        #     base_model.add(Activation('relu'))
        #     base_model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        #     base_model.add(Conv2D(64, (3, 3), padding='same'))
        #     base_model.add(Activation('relu'))
        #     base_model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        #     base_model.add(Conv2D(64, (3, 3), padding='same'))
        #     base_model.add(Activation('relu'))
        #     base_model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))
        #     base_model.add(Flatten())

        #     x = base_model.output
        #     x = GlobalAveragePooling2D()(x)
        #     predictions = Dense(6, activation="linear")(x)
        #     model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])            
        
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        # with self.graph.as_default():
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        # with self.graph.as_default():
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, \
                       callbacks=[self.tensorboard_callback])

        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, NUMBER_OF_ACTIONS)).astype(np.float32)
        self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == '__main__':
    FPS = 60
    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.keras.utils.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:
            env.collision_hist = []

            # Update tensorboard step every episode
            # agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            # print("==== get_actors ====")
            # print(env.world.get_actors().filter("*vehicle*"))
            current_state = env.reset()
            # print(env.world.get_actors().filter("*vehicle*"))

            # Reset flag and start iterating until episode ends
            done = False
            episode_start = time.time()

            # Play for given number of seconds only
            while True:
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, NUMBER_OF_ACTIONS)
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/FPS)

                new_state, reward, done, _ = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1

                if SHOW_PREVIEW and episode % SHOW_PREVIEW_EVERY == 0:
                    images = np.concatenate((env.rgb_sensor_data, env.sem_sensor_data), axis=1)
                    cv2.imshow("sensor images", images)
                    cv2.waitKey(10)
                              
                if done:
                    break

            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.model')

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

            # print status
            print("episode=", episode, ", reward=", episode_reward, ", epsilon=", epsilon, ", collision=", len(env.collision_hist))


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.model')