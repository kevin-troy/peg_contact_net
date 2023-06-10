import numpy as np
from scenario import gen_scenario, gen_peg_profile, gen_hole_profile
from sensor import gen_sensor_reading

if __name__ == "__main__":

    n_scenarios = 100
    n_meas_per_scenario = 10
    res = 10000
    theta_bounds = [-np.pi/4, np.pi/4]

    # 1.) Define the peg and hole parameters
    hole = gen_hole_profile(resolution=res)
    peg, peg_corners, peg_sides, peg_params = gen_peg_profile(resolution=res)

    # 2.) Generate scenarios and sensed wrenches
    x_data = []
    y_data = []
    thetas = []

    scenario_counts = 0
    while scenario_counts < n_scenarios:

        # Generate a random theta, scenario, and sensor readings
        theta = np.random.uniform(theta_bounds[0], theta_bounds[1])
        scenario = gen_scenario(peg, hole, peg_corners, theta, res, multi_contact=True)

        if scenario["contacts_peg"].shape[1] != 2:
            print("Invalid scenario...omitting, iter=", len(x_data))
            continue

        sensor_counts = 0
        while sensor_counts < n_meas_per_scenario:
            wrenches, valid = gen_sensor_reading(peg_params, theta, scenario)
            if valid:
                thetas.append(theta)
                y_data.append(np.hstack(scenario["contacts_peg"]))
                x_data.append(wrenches)
                sensor_counts +=1
        scenario_counts +=1

    x_data = np.vstack(x_data) # rows = scenario number, cols = Fx, Fy, Mz
    y_data = np.vstack(y_data) # rows = scenario number, cols = x1, y1, x2, y2
    data = np.hstack([x_data, y_data, np.vstack(thetas)])
    np.savetxt("data_val_small.csv", data, delimiter=",")
