import numpy as np
import numpy.random as rn


def _within_left_sensor(dw):
    return (np.pi / 8) <= dw < (np.pi / 2)


def _within_right_sensor(dw):
    return (-np.pi / 2 < dw <= -np.pi / 8)


class Environment(object):

    def __init__(self, no_of_objects, min_size, max_size, x_max, y_max):

        self.x_max = x_max
        self.y_max = y_max

        self.min_size = min_size
        self.max_size = max_size

        self.objects = [self._new_object() for _ in range(no_of_objects)]

    def read_sensors(self, x, y, w):
        """
        Computes sensor readings for a robot positoned at (x,y) with orientation of w inside the
        environment.
        """

        sensor_left = 0
        sensor_right = 0

        # Run through objects and take the largest stimulus as sensor reading
        for obj in self.objects:
            sl, sr = self._sensor_reading_from_object([x, y, w], [obj['x'], obj['y']])
            sensor_left, sensor_right = max(sensor_left, sl), max(sensor_right, sr)

        return sensor_left, sensor_right

    def _new_object(self):
        return {'x': rn.rand() * self.x_max,
                'y': rn.rand() * self.y_max,
                'r': self.min_size + rn.rand() * (self.max_size - self.min_size)}

    def _sensor_reading_from_object(self, (x, y, w), (ox, oy)):
        """
        Determines the individual sensor readings from the given co-ords to the object
        """

        sensor_range = 20.0
        dx, dy, z = self._distance_from([x, y], [ox, oy])

        if z < sensor_range:
            v = np.arctan2(dx, dy)
            dw = v - w  # difference in robot's heading and object
            stimulus = (sensor_range - z) / sensor_range

            if (np.pi / 8) <= abs(dw) < (np.pi / 2):
                if dw < 0:
                    return stimulus, 0
                else:
                    return 0, stimulus

        return 0, 0

    def _distance_from(self, (x1, y1), (x2, y2)):
        """
        The computed distance is the closest across the torus. This means we will travel across the
        boundary if that is a closer match.
        """

        x2 = x2 + self.x_max if abs(x2 + self.x_max - x1) < abs(x2 - x1) else x2
        y2 = y2 + self.y_max if abs(y2 + self.y_max - y1) < abs(y2 - y1) else y2

        dx = x2 - x1
        dy = y2 - y1

        z = np.sqrt(dx ** 2 + dy ** 2)

        return dx, dy, z
