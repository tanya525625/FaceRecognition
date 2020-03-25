import random


def generate_random_points(img_shape, poins_count, epsilon=40):
    coord_x = []
    coord_y = []
    x0_values = [0, 50, 0, 0]
    k_values = [1, -1, 0.1, 10]
    for i in range(poins_count):
        x = random.randint(0, img_shape[0] - 1)
        for k, x0 in zip(k_values, x0_values):
            y = find_point_on_line(x, k, x0)
            if y - epsilon < 0 or y + epsilon < 0:
                continue
            else:
                y = random.randint(int(y - epsilon), int(y + epsilon))
                if y < img_shape[1]:
                    coord_y.append(y)
                    coord_x.append(x)

    return list(zip(coord_x, coord_y))


def find_point_on_line(x, k, x_0=0):
    return x_0 + k*x


def random_method(img, coordinate_points):
    return [img[point[0]][point[1]][color] for point in coordinate_points for color in range(3)]
