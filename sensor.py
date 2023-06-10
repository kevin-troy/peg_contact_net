"""
sensor.py - Functionality for converting contact scenarios into sensor measurements.
"""
import numpy as np
from scenario import rot, gen_scenario, gen_peg_profile, gen_hole_profile, plot_scenario, plot_scenario_peg_frame
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint


def gen_sensor_reading(peg_params, theta, scenario, debug=False, mu=0.1, use_default_formulation=True):
    x_basis = np.array([1., 0.])
    y_basis = np.array([0., 1.])
    z_basis = np.array([0., 0., 1.])
    # First, determine gravity vector and force application vector in the peg frame
    fg = peg_params["mass"]*9.81*rot(-theta) @ -y_basis

    # next, generate max friction basis vecs
    cvs = scenario["contact_vecs_peg"]
    friction_bases = [rot(-np.pi/2)@basis for basis in cvs]

    #friction_bases = []
    for idx, basis in enumerate(friction_bases):
        if np.dot(basis, y_basis) < 0:
            pass #TODO: see if this is needed
            #friction_bases[idx] = rot(np.pi)@basis
    friction_bases = [basis/np.linalg.norm(basis) for basis in friction_bases]
    contacts_peg = scenario["contacts_peg"]

    if debug:
        # Compute dot products, should be zero
        dot_check = []
        for idx, base in enumerate(friction_bases):
            dot_check+=[np.dot(base, cvs[idx])]

        assert (np.abs(dot_check) <= 1e-6).all()

        arrow_scale = 3
        plot_scenario_peg_frame(scenario, theta, fg/np.linalg.norm(fg))
        plt.arrow(0, 0, fg[0]/np.linalg.norm(fg), fg[1]/np.linalg.norm(fg), head_width=.1, color="red", label="Gravity Vector")
        for i in range(contacts_peg.shape[1]):
            plt.arrow(contacts_peg[0,i], contacts_peg[1,i], friction_bases[i][0]/arrow_scale, friction_bases[i][1]/arrow_scale, head_width=0.05,
                     color="brown", length_includes_head=True)
            plt.scatter(contacts_peg[0,i], contacts_peg[1,i])

        plt.show()

    r_sensor_to_cm = rot(theta) @ -y_basis * peg_params["height"]
    frame_bases = [x_basis, y_basis]


    # Next, formulate the optimization problem(s) to solve for the minimum and maximum applied wrenches
    def contact_function(x, fg, contacts_peg, friction_bases, normal_bases, r_sensor_to_cm):
        # We want to minimize the sum of force and torques (ideally zero), given the contact locations and directions.
        # Lets do this in the peg frame, and solve for scalars that will define the norms of the applied forces and
        # Init wrench sum with robot wrench
        force = x[0]*r_sensor_to_cm/np.linalg.norm(r_sensor_to_cm)+fg
        moment = x[1]
        scale_idx = 2

        # Iterate over all contact locations
        for contact_loc, normal_force, friction_base in zip(contacts_peg, normal_bases, friction_bases):
            n_vec = x[scale_idx]*(normal_force - contact_loc)
            f_vec = x[scale_idx+1]*(friction_base - contact_loc)
            moment += np.cross(contact_loc + r_sensor_to_cm, n_vec + mu * f_vec)
            force += n_vec + mu * f_vec
            scale_idx += 2
        wrench_sum = (force**2).sum() + (moment**2)
        return wrench_sum

    x0 = 0.*np.ones((len(contacts_peg)*2+2,1))
    #x0 = np.random.randint(-10, 10, (len(contacts_peg)*2+2,1))
    #_=contact_function(x0, fg, contacts_peg, friction_bases, cvs, r_sensor_to_cm)

    # Define friction cone constraints. i.e. the magnitude of the friction forces must be less than the friction coef
    # times the associated normal force.
    # Recall that the first two elements of the x vector are the robot force and torque. Thus each 2n:2n+1 for n>=1
    # correspond to the normal and friction magnitudes, respectively.
    if contacts_peg.shape[1] != 2 or len(cvs) != 2:
        contacts_peg = contacts_peg[:, :2]
        cvs = cvs[:2]

    constraints = []
    for n in range(len(cvs)):
        # Constraint for friction cones
        con = lambda x: np.abs(x[2 * n + 2]) - np.abs(
            x[2 * n + 1] * mu)  # abs to make this constraint invariant to direction
        constraints += [NonlinearConstraint(con, -np.inf, 0.)]
        # Constraint to keep normal forces positive (prevents changing direction)
        con = lambda x: x[2 * n + 2]
        constraints += [NonlinearConstraint(con, 0., np.inf)]


    """
    Problem formulations:
    
    Formulation 01: Attempts to find values for the robot force and torque that minimize the contact function. This is 
    essentially a spaghettified root-finding problem, were we use the randomness in the initial parameters to induce 
    various solutions that still agree with the wrench sum == 0
    
    Formulation 02: Since this is an inequality problem, this formulation aims to enforce the wrench sum == 0 as a 
    constraint and directly minimizes the robot wrench. To provide various solutions, wrenchs are sampled from between 
    the minimum and maximum wrenches.  
    
    WIP, when using no constraints, this seems to work fine and gives the desired 
    symmetry, but with constraints seems to blow up. There are some cases where the normal forces "blow up" the maximum 
    force. i.e when the two contact vectors are orthogonal, the entirety of the wrench space is spanned, and infinite 
    robot wrench can be applied.
    
    """

    if use_default_formulation:
        x0 = np.random.randint(-5, 5, (len(contacts_peg)*2+2,1))
        #x0 = np.zeros((len(contacts_peg) * 2 + 2, 1))
        sol = minimize(contact_function, x0, args=(fg, contacts_peg, friction_bases, cvs, r_sensor_to_cm), constraints=constraints)
        robot_wrench = np.concatenate([sol.x[0]*r_sensor_to_cm/np.linalg.norm(r_sensor_to_cm), [sol.x[1]]])
        is_valid = sol.success and np.abs(sol.fun)<1e-6
        return robot_wrench, is_valid
    else:
        x0 = 0. * np.ones((len(contacts_peg) * 2 + 2, 1))
        # Alternate formulation. Minimize and maximize |F_robot|, |M_robot|
        def min_objective(x):
            return np.sum(x[:3])

        def max_objective(x):
            return -np.sum(x[:3]**2)

        con = lambda x: contact_function(x, fg, contacts_peg, friction_bases, cvs, r_sensor_to_cm)
        constraints += [NonlinearConstraint(con, 0., 0.)]
        sol_min = minimize(min_objective, x0, constraints=constraints)
        #sol_max = minimize(max_objective, x0, constraints=constraints)
        # TODO: Sample between min and max objectives to span equality range
        sol = sol_min
        return np.concatenate([sol.x[0]*r_sensor_to_cm/np.linalg.norm(r_sensor_to_cm), [sol.x[1]]]), sol.success

if __name__ == "__main__":

    hole = gen_hole_profile(resolution=10000)
    peg, peg_corners, peg_sides, peg_params = gen_peg_profile(resolution=10000)

    theta = np.linspace(-np.pi/8, np.pi/8, 100)
    wrenches = []
    for th in theta:
        scenario = gen_scenario(peg, hole, peg_corners, th, resolution=10000, multi_contact=True, debug_plots=False)
        wrenches += [gen_sensor_reading(peg_params, th, scenario)[0]]

    wrenches = []
    th = np.pi/8
    scenario = gen_scenario(peg, hole, peg_corners, th, resolution=10000, multi_contact=True, debug_plots=False)
    for i in range(100):
        wrenches += [gen_sensor_reading(peg_params, th, scenario)[0]]
    print("here")
    wrenches = np.vstack(wrenches)
    plt.figure()
    labels = ["Fx", "Fy", "Mz"]
    for i in range(3):
        plt.plot(wrenches[:,i], '-o', label=labels[i])
    plt.grid()
    plt.xlabel("Theta (rad)")
    plt.ylabel("Sensor Reading")

    plt.figure()

    force_applied = np.linalg.norm(wrenches[:,:2], axis=1)
    plt.plot(theta, force_applied)

    # Investigate symmetry of results.
    plt.figure()
    pivot = wrenches.shape[0]/2
    plt.plot(theta[:50],wrenches[:50,0])
    plt.plot(theta[:50],np.flip(wrenches[50:,0],axis=0))

    plt.figure()
    plt.plot(theta[:50],wrenches[:50,1])
    plt.plot(theta[:50],-np.flip(wrenches[50:,1],axis=0))



    plt.show()

