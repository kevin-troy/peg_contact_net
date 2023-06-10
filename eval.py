import numpy as np
import torch
import matplotlib.pyplot as plt
from scenario import gen_scenario, gen_peg_profile, gen_hole_profile, rot
from sensor import gen_sensor_reading
from contact_net import ContactNet


def eval_plots(model=None):
    # Init scenario
    resolution=10000
    theta = np.linspace(-np.pi/4, np.pi/4, 100)
    hole = gen_hole_profile(resolution=resolution)
    peg, peg_corners, _, peg_params = gen_peg_profile(resolution=resolution)

    all_y = []
    all_yhat = []

    for th in theta:
        # Gen scenario for this theta
        scenario_valid = False

        while not scenario_valid:
            scenario = gen_scenario(peg, hole, peg_corners, th, resolution, multi_contact=True,
                                    debug_plots=False)
            scenario_valid = scenario["contacts_peg"].shape[1] == 2
        # Generate sensor reading
        wrench_valid = False
        while not wrench_valid:
            wrench, wrench_valid = gen_sensor_reading(peg_params, th, scenario, mu=0.1)

        # Evaluate network
        y_est = model(torch.Tensor(np.array([wrench]))).detach().numpy()
        y_true = np.hstack([scenario["contacts_peg"].reshape(-1,4), [[th]]])
        all_y.append(y_true)
        all_yhat.append(y_est)

    all_y = np.vstack(all_y)
    all_yhat = np.vstack(all_yhat)

    # Plot 1.) - True v. Est locations
    plt.figure()
    labels = ["x1", "x2", "y1", "y2"]
    colors = ["blue", "red", "orange", "black"]
    for i in range(4):
        plt.plot(theta, all_y[:,i], color = colors[i], label=labels[i]+" true")
        plt.plot(theta, all_yhat[:, i], color = colors[i], ls="--", marker="*", label=labels[i]+" est.")
    plt.xlabel("Theta (rad)")
    plt.ylabel("Contact Component (in)")
    plt.grid()
    plt.title("True v. Est. Contact Locations")
    plt.legend()


    # Plot 2.) True v. est locations error plot
    plt.figure()
    for i in range(4):
        plt.plot(theta, (all_y[:,i]-all_yhat[:,i]), marker="o", label=labels[i]+"Error")
    plt.grid()
    plt.legend()
    plt.xlabel("Theta (rad)")
    plt.ylabel("L1 Contact Component Error (in)")
    plt.title("Contact Component Errors")


    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(theta, all_yhat[:,-1], "-*", label="Est. Theta")
    plt.plot(theta, theta, label="True Theta")
    plt.xlabel("Theta (rad)")
    plt.ylabel("Theta (rad)")
    plt.grid()
    plt.title("Angular Pose Results")

    plt.subplot(2,1,2)
    plt.plot(theta, (theta-all_yhat[:,-1]))
    plt.xlabel("Theta (rad)")
    plt.ylabel("Theta Est. Error (rad)")
    plt.grid()

    # Scatter hole estimate NOTE: cool idea, but needs some knowledge of x, y
    """
    plt.figure()
    plt.title("Hole Geometry Reconstruction")
    y_inertial = []
    for y_hat, th in zip(all_yhat, theta):
        y_inertial.append(np.hstack([rot(th)@y_hat[:2], rot(th)@y_hat[2:-1]]))
    # TODO: use estimated theta and contact locations to scatter to the hole. PLot hole over this and see how well
    # it is "mapped"
    y_inertial=np.vstack(y_inertial)
    plt.scatter(y_inertial[:,0], y_inertial[:,2])
    plt.scatter(y_inertial[:,1], y_inertial[:,3])
    """
    plt.show()
    print("here")

def eval_gif(model=None):
    from matplotlib.animation import FuncAnimation
    n_frames = 300
    theta = np.linspace(-np.pi/4, np.pi/4, n_frames)
    multi_contact = True
    fig, axs = plt.subplots(1,2)
    def animate(i):
        y_basis = rot(-theta[i])@np.array([0,-1])
        resolution = 10000
        ax = axs[0]
        hole = gen_hole_profile(resolution=resolution)
        peg, peg_corners, _, peg_params = gen_peg_profile(resolution=resolution)
        scenario = gen_scenario(peg, hole, peg_corners, theta[i], resolution, multi_contact=multi_contact, debug_plots=False)
        ax.clear()
        peg = scenario["peg"]
        hole = scenario["hole"]
        peg_corners = scenario["corners"]
        contacts_inertial = scenario["contacts_inertial"]
        contacts_peg = scenario["contacts_peg"]
        contact_vecs = scenario["contact_vecs_inertial"]
        contact_vecs_peg = scenario["contact_vecs_peg"]

        ax.plot(peg[0,:], peg[1,:])
        ax.plot(hole[0,:], hole[1,:])
        for cid in range(contacts_inertial.shape[1]):
            ax.scatter(contacts_inertial[0,cid], contacts_inertial[1,cid], marker="*", label="Contact "+str(cid))
            sc = ax.scatter(contacts_inertial[0, cid], contacts_inertial[1, cid], marker="o", s=500)
            sc.set_facecolor("none")
            sc.set_edgecolor("black")

            arrow_scale = 2
            x0 = contacts_inertial[0, cid] - contact_vecs[cid][0] / arrow_scale
            y0 = contacts_inertial[1, cid] - contact_vecs[cid][1] / arrow_scale
            ax.arrow(x0, y0, contact_vecs[cid][0] / arrow_scale, contact_vecs[cid][1] / arrow_scale, head_width=0.05,
                      color="red", length_includes_head=True)

            ax.legend()
        ax.fill_between(hole[0, :], hole[1, :], -2 * np.ones_like(hole[1, :]), color="orange", alpha=0.5, hatch="/", edgecolor="orange")
        ax.fill(peg_corners[0,:], peg_corners[1,:], color="blue", alpha=0.2)

        ax.set_aspect("equal")
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.grid(alpha=0.2)
        ax.set_xlabel("Inertial X")
        ax.set_ylabel("Inertial Y")
        ax.set_title("Contact Scenario vs. Angle")

        # Second subplot
        # Evaluate the model to get the contact point estimates.
        wrench_valid=False
        while not wrench_valid:
            wrench, wrench_valid = gen_sensor_reading(peg_params, theta[i], scenario, mu=0.1)
        y = model(torch.Tensor(np.array([wrench]))).detach().numpy()
        contacts_est = y[:,:4].reshape(-1,2)
        theta_est = y[:,-1]
        ax = axs[1]
        ax.clear()

        contacts_peg = scenario["contacts_peg"]
        peg = rot(-theta[i])@scenario["peg"]
        corners = rot(-theta[i]) @ scenario["corners"]
        delta = np.mean(corners, axis=1)

        ax.plot(peg[0, :]-delta[0], peg[1, :]-delta[1])
        ax.scatter(contacts_peg[0, :], contacts_peg[1, :])
        ax.fill(corners[0, :]-delta[0], corners[1, :]-delta[1], color="blue", alpha=0.2, hatch="/", edgecolor="blue")
        ax.arrow(0,0,y_basis[0], y_basis[1], head_width=.1, color="red", label="Gravity Vector")

        for cid in range(contacts_inertial.shape[1]):
            arrow_scale = 2
            x0 = contacts_peg[0, cid] - contact_vecs_peg[cid][0] / arrow_scale
            y0 = contacts_peg[1, cid] - contact_vecs_peg[cid][1] / arrow_scale
            ax.arrow(x0, y0, contact_vecs_peg[cid][0] / arrow_scale, contact_vecs_peg[cid][1] / arrow_scale, head_width=0.05,
                      color="red", length_includes_head=True)
        ax.scatter(contacts_est[0,:], contacts_est[1,:], marker="*", label="Est Contacts")

        ax.legend()
        ax.set_aspect("equal")
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.grid(alpha=0.2)
        ax.set_xlabel("Peg X")
        ax.set_ylabel("Peg Y")
        ax.set_title("Contacts in Peg Frame")

        fig.tight_layout()
    ani = FuncAnimation(fig, animate, frames=n_frames, interval=0.01, repeat=True)
    plt.show()

def loss_plot(file_base):
    import pandas as pd
    df = pd.read_csv("./results/"+file_base+".csv")

    plt.figure()
    plt.plot(df["train_loss"], label="Train Loss")
    plt.plot(df["val_loss"], '--', label="Val. Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.title("Training Losses for model="+file_base)
    plt.show()



if __name__ == "__main__":
    use_clip = True
    use_small = True
    file_base = "contact_net_v1"

    if use_clip:
        file_base+="_clipped"
    else:
        file_base+="_unclipped"

    if use_small:
        file_base+="_small"

    if use_clip:
        mdl = ContactNet(peg_width=0.75, peg_height=1.5)
        mdl.load_state_dict(torch.load("./models/"+file_base+".pt"))
    else:
        mdl = ContactNet()
        mdl.load_state_dict(torch.load("./models/"+file_base+".pt"))
    mdl.eval()
    #eval_plots(mdl)
    #eval_gif(mdl)
    loss_plot(file_base)