import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.signal import find_peaks


def gen_hole_profile(width=1, height=1, resolution=1000):
    side_x = np.linspace(-width/2, width/2, int(width*resolution))
    side_y = np.linspace(0, height, int(height*resolution))
    hole_x = np.concatenate([side_x-width, -0.5*width*np.ones_like(side_y), side_x, 0.5*width*np.ones_like(side_y), side_x+width], axis=-1)
    hole_y = np.concatenate([np.zeros_like(side_x), -side_y, -height*np.ones_like(side_x), side_y-height, np.zeros_like(side_x)], axis=-1)
    return np.vstack([hole_x, hole_y])


def gen_peg_profile(width=0.75, height=1.5, mass=1, resolution=1000):
    side_y = np.linspace(-height/2, height/2, int(height*resolution))
    side_x = np.linspace(-width/2, width/2, int(width*resolution))
    peg_x = np.concatenate([side_x[0]*np.ones_like(side_y), side_x, side_x[-1]*np.ones_like(side_y), -side_x], axis=-1)
    peg_y = np.concatenate([side_y, side_y[-1]*np.ones_like(side_x), -side_y, side_y[0]*np.ones_like(side_x)], axis=-1)
    corners = np.array([[-width/2, -width/2, width/2,  width/2], [-height/2, height/2, height/2, -height/2]])

    sides = [np.vstack([side_x[0]*np.ones_like(side_y),side_y]), np.vstack([side_x, side_y[-1]*np.ones_like(side_x)]),
             np.vstack([side_x[-1]*np.ones_like(side_y), -side_y]), np.vstack([-side_x, side_y[0]*np.ones_like(side_x)])]
    peg_params = {"width": width, "height": height, "mass": mass}
    return np.vstack([peg_x, peg_y]), corners, sides, peg_params


def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])


def get_contact_profiles(peg_corners, hole):
    # To properly interp, trip out the top corner.
    top_corner = np.argmax(peg_corners[1,:])
    peg_bottom_profile = np.delete(peg_corners, top_corner, axis=1)

    # Sort by x value
    peg_bottom_profile = peg_bottom_profile[:, np.argsort(peg_bottom_profile, axis=1)[0, :]]

    # contact_profile_hole = projection of peg bottom face(s) onto hole.
    contact_profile_hole = hole[:,(peg_bottom_profile[0,0] < hole[0,:]) & (hole[0,:] < peg_bottom_profile[0,-1])]

    # contact_profile_peg_y = bottom face(s) of peg
    contact_profile_peg_y = np.interp(contact_profile_hole[0,:], peg_bottom_profile[0,:], peg_bottom_profile[1,:])

    # Determine interference profile and find maximum interference.
    interference = contact_profile_hole[1,:] - contact_profile_peg_y
    contact_ids, _ = find_peaks(interference)

    contact_profile_peg = np.vstack([contact_profile_hole[0, :], contact_profile_peg_y])

    return interference, contact_ids, contact_profile_hole, contact_profile_peg



def shift(theta, peg, peg_corners, hole, contact_profile_peg, contact_profile_hole, plots=True, vertical=False):
    if vertical:
        R = rot(-theta)
        peg = R@peg
        peg_corners = R@peg_corners
        hole = R@hole
        contact_profile_peg = R@contact_profile_peg
        contact_profile_hole = R@contact_profile_hole
    else:
        R = rot(np.sign(theta)*np.pi/2)
        peg = R@peg
        peg_corners = R@peg_corners
        hole = R@hole
        contact_profile_peg = R@contact_profile_peg
        contact_profile_hole = R@contact_profile_hole
    if plots:
        plt.figure()
        plt.plot(peg[0, :], peg[1, :])
        plt.scatter(peg_corners[0, :], peg_corners[1, :])
        plt.scatter(contact_profile_peg[0,:], contact_profile_peg[1,:])
        plt.scatter(contact_profile_hole[0, :], contact_profile_hole[1, :])
        plt.plot(hole[0, :], hole[1, :])
        plt.axis("equal")
        plt.title("Second Rotation")

        # Now that we are aligned with the peg instead, move "horizontal"
        # until contact is forced again.
        plt.figure()
        plt.plot(peg[0, :], peg[1, :])
        plt.scatter(peg_corners[0, :], peg_corners[1, :])
        plt.scatter(hole[0, :], hole[1, :])
        plt.axis("equal")

    # Now, trim to only elements of the contact profile that have the same X.
    # Project in rotated space to find potential contacts
    tol = 1e-12
    mask = np.abs(contact_profile_peg[0,:]-scipy.stats.mode(contact_profile_peg[0,:],keepdims=False)[0])>tol
    contact_profile_peg_side = contact_profile_peg[:,mask]

    # In order for interp to work in this transformed workspace, we must ensure there are no points with duplicate y values.
    # This can be done by recognizing that if theta is positive (CCW), we want to look at the left half of the hole profile.
    # If theta is negative, we want to look at the right half of the hole profile.
    # If we are looking at a vertical shift, no trimming is necessary as we are looking to make contact with the hole's bottom.
    idx = int(hole.shape[1]/2)
    if not vertical:
        if theta > 0:
            contact_profile_hole_side_y = np.interp(contact_profile_peg_side[0,:], hole[0,:idx], hole[1,:idx])
        else:
            contact_profile_hole_side_y = np.interp(contact_profile_peg_side[0,:], hole[0,idx:], hole[1,idx:])
    else:
        contact_profile_hole_side_y = np.interp(contact_profile_peg_side[0, :], hole[0, :], hole[1, :])
    contact_profile_hole_side = np.vstack([contact_profile_peg_side[0,:],contact_profile_hole_side_y])

    if plots:
        plt.scatter(contact_profile_peg_side[0,:], contact_profile_peg_side[1,:])
        plt.scatter(contact_profile_hole_side[0, :], contact_profile_hole_side[1, :])
        plt.title("Second Rotation - Modified Profiles")

    # Finally, add the "vertical" offset to provide another potential contact.
    interference = contact_profile_peg_side[1,:] - contact_profile_hole_side[1,:]
    peg[1,:] -= np.min(np.abs(interference))
    peg_corners[1,:] -= np.min(np.abs(interference))
    contact_profile_peg[1,:] -= np.min(np.abs(interference))

    if plots:
        plt.figure()
        plt.plot(peg[0, :], peg[1, :])
        plt.scatter(peg_corners[0, :], peg_corners[1, :])
        plt.plot(hole[0, :], hole[1, :])
        plt.axis("equal")
        plt.title("Final Placement (detailed+rotated)")

    # Finally, unrotate to nominal
    if vertical:
        R = rot(theta)
        peg = R@peg
        peg_corners = R@peg_corners
        hole = R@hole
        contact_profile_peg = R@contact_profile_peg
        contact_profile_hole = R@contact_profile_hole
    else:
        R = rot(-np.sign(theta)*np.pi/2)
        peg = R@peg
        peg_corners = R@peg_corners
        hole = R@hole
        contact_profile_peg = R@contact_profile_peg
        contact_profile_hole = R@contact_profile_hole
    return peg, peg_corners, hole, contact_profile_peg, contact_profile_hole


def get_contact_vectors(hole, theta, contact_locations):
    # Uses knowledge of the hole geometry and inertial contact locations to determine contact vectors
    # Deals with three primary cases:
    # 1.) Horizontal contact vectors,
    # 2.) Vertical contact vectors,
    # 3.) Corner contact vectors - for which theta is needed

    # Check for contacts at each of 5 faces:
    #-+ +-   In this terrible diagram, each face 0-4 is shown with corners.
    # |_|    Corner contacts are determined via multiple face contacts.

    # Use bounds of x, y to infer geometry
    width = (np.max(hole[0,:])-np.min(hole[0,:]))/3
    height = (np.max(hole[1,:])-np.min(hole[1,:]))

    # Define potential contact vectors for all 5 sides and two upper corners.
    contact_bases = [np.array([0.,1.]),
                     np.array([1.,0.]),
                     np.array([0.,1.]),
                     np.array([-1.,0.]),
                     np.array([1., 0.])]

    contact_bases_peg = [rot(-theta) @ np.array([0., 1.]),
                         rot(-theta) @ np.array([1., 0.]),
                         rot(-theta) @ np.array([0., 1.]),
                         rot(-theta) @ np.array([-1., 0.]),
                         rot(-theta) @ np.array([1., 0.])]

    # Fix some domain issue (?) with rotation matrices
    if theta >=0:
        contact_bases += [rot(theta-np.pi/2)@np.array([0., 1.])]
        contact_bases += [rot(theta)@np.array([0., 1.])]
        contact_bases_peg += [np.array([1., 0.])]
        contact_bases_peg += [np.array([0.,1.])]
    else:
        contact_bases += [rot(theta)@np.array([0., 1.])]
        contact_bases += [rot(theta+np.pi/2)@np.array([0., 1.])]
        contact_bases_peg += [np.array([0., 1.])]
        contact_bases_peg += [np.array([-1.,0.])]

    def isclose(x, y, tol=1e-3):
        return np.abs(x - y) < tol

    face_contacts = []
    contact_vecs = []
    contact_vecs_peg = []

    for i in range(contact_locations.shape[1]):
        # Face 0 contact - left half of hole and approx. zero height.
        face_contacts.append([])
        face_contacts[i] += [isclose(contact_locations[1,i], 0.) and (contact_locations[0,i] < 0.)]
        # Face 1 contact - approx x=-width/2
        face_contacts[i] += [isclose(contact_locations[0,i], -width/2)]
        # Face 2 contact - -width/2<=x<=width/2 and height approx -height
        face_contacts[i] += [(-width/2<=contact_locations[0,i]<=width/2) and isclose(contact_locations[1,i], -height)]
        # Face 3 contact - approx x=width/2
        face_contacts[i] += [isclose(contact_locations[0,i], width/2)]
        # Face 4 contact - right half of hole and approx zero height
        face_contacts[i] += [isclose(contact_locations[1,i], 0.) and (contact_locations[0,i]>0.)]

        # Now that face contacts have been defined, check for corners.
        if np.sum(face_contacts[i]) == 1.:
            contact_vecs += [contact_bases[face_contacts[i].index(True)]]
            contact_vecs_peg += [contact_bases_peg[face_contacts[i].index(True)]]
        elif np.sum(face_contacts[i]) == 2.:
            if np.array(face_contacts[i][-2:]).all():
                # Right corner contact
                contact_vecs += [contact_bases[6]]
                contact_vecs_peg += [contact_bases_peg[6]]
            elif np.array(face_contacts[i][:2]).all():
                contact_vecs += [contact_bases[5]]
                contact_vecs_peg += [contact_bases_peg[5]]

        else:
            raise ValueError("unclassifiable contact scenario detected")

    return contact_vecs, contact_vecs_peg, face_contacts


def gen_scenario(peg, hole, peg_corners, theta, resolution=1000, multi_contact=False, debug_plots=False):
    R = rot(theta)
    """
    GENERATE INITIAL CONTACT
    """
    peg = R@peg
    peg_corners = R@peg_corners
    interference, contact_ids, _, _ = get_contact_profiles(peg_corners, hole)
    max_interference = np.argmax(interference)

    # Shift peg to include max interference, resulting in only a single contact.
    peg[1,:] += interference[max_interference]
    peg_corners[1,:] += interference[max_interference]

    if debug_plots:
        plt.figure()
        plt.plot(peg[0,:], peg[1,:])
        plt.scatter(peg_corners[0,:], peg_corners[1,:])
        plt.plot(hole[0,:], hole[1,:])
        plt.axis("equal")
        plt.title("Initial Rotation")

    # Generate interference
    interference, contact_ids, contact_profile_hole, contact_profile_peg = get_contact_profiles(peg_corners, hole)

    if debug_plots:
        plt.figure()
        plt.plot(peg[0,:], peg[1,:])
        plt.scatter(peg_corners[0,:], peg_corners[1,:])
        plt.plot(hole[0,:], hole[1,:])
        plt.axis("equal")
        #for i in range(4):
            #plt.plot(peg_sides[i][0,:], peg_sides[i][1,:], label=str(i))
        plt.legend()
        plt.title("Initial Rotation - reversed")

    """
    Attempt to enforce additional contacts by "sliding" the peg in multiple directions.
    """
    if multi_contact:
        # HORIZONTAL SHIFT
        peg, peg_corners, hole, contact_profile_peg, contact_profile_hole = shift(theta, peg, peg_corners, hole, contact_profile_peg, contact_profile_hole, plots=False)
        # VERTICAL SHIFT
        peg, peg_corners, hole, contact_profile_peg, contact_profile_hole = shift(theta, peg, peg_corners, hole, contact_profile_peg, contact_profile_hole, vertical=True, plots=False)
        # HORIZONTAL SHIFT NO. 2
        peg, peg_corners, hole, contact_profile_peg, contact_profile_hole = shift(theta, peg, peg_corners, hole, contact_profile_peg, contact_profile_hole, plots=False)

    # Get final interference profile(s)
    _, _, final_hole, final_peg = get_contact_profiles(peg_corners, hole)
    interference = np.linalg.norm(final_hole-final_peg, axis=0)

    # intermediate shift for interference to find peaks
    delta = np.max(interference)
    interference = -interference + delta

    # minimum spacing of 0.1 (must convert to resolution) - prevents duplicates
    contact_ids,_ = find_peaks(interference, distance=0.1*resolution)
    # undo intermediate shift
    interference = interference - delta

    # Trim out any peaks that do not show contact lower than threshold (0 distance = contact)
    contact_ids = contact_ids[np.abs(interference[contact_ids]) < 1e-2]

    if debug_plots:
        plt.figure()
        plt.plot(interference)
        plt.scatter(contact_ids, interference[contact_ids])

        plt.figure()
        plt.plot(peg[0,:], peg[1,:])
        plt.scatter(peg_corners[0,:], peg_corners[1,:])
        plt.plot(hole[0,:], hole[1,:])
        plt.axis("equal")
        plt.scatter(final_peg[0, contact_ids], final_peg[1,contact_ids], marker="*")
        plt.title("final placement")

    contacts_inertial = np.vstack([final_peg[0, contact_ids], final_peg[1,contact_ids]])

    contacts_peg = rot(-theta)@contacts_inertial
    delta = np.mean(rot(-theta)@peg_corners, axis=1)
    contacts_peg[0,:] -= delta[0]
    contacts_peg[1,:] -= delta[1]

    # Finally, use knowledge of the simplified hole geometry to determine
    # contact force vectors directions. Anchor these with the contact points themselves
    # to define the basis-es (lol? bases sounds wrong)
    contact_vecs, contact_vecs_peg, face_contacts = get_contact_vectors(hole, theta, final_hole[:,contact_ids])

    #contact_vecs_peg = [rot(-theta)@vec-delta for vec in contact_vecs]


    scenario = {"peg":peg,
                "corners":peg_corners,
                "hole":hole,
                "contacts_inertial":contacts_inertial,
                "contacts_peg":contacts_peg,
                "contact_vecs_inertial": contact_vecs,
                "contact_vecs_peg": contact_vecs_peg}
    if len(contact_vecs) != contact_ids.size:
        print("[SCENARIO]: Unexpected size mismatch between contact vectors and contact ids")

    """
    for face_contact in face_contacts:
        if np.sum(face_contact)>0:
            print(face_contact)
            print(face_contacts)
            plot_scenario(scenario)
            plt.show()
    """
    return scenario

def plot_scenario(scenario):
    peg = scenario["peg"]
    hole = scenario["hole"]
    peg_corners = scenario["corners"]
    contacts_inertial = scenario["contacts_inertial"]
    contact_vecs = scenario["contact_vecs_inertial"]

    plt.figure()
    plt.plot(peg[0, :], peg[1, :])
    plt.scatter(peg_corners[0, :], peg_corners[1, :])
    plt.plot(hole[0, :], hole[1, :])
    plt.axis("equal")
    for i in range(contacts_inertial.shape[1]):
        plt.scatter(contacts_inertial[0,i], contacts_inertial[1,i], marker="*", label="Contact "+str(i))
        sc = plt.scatter(contacts_inertial[0, i], contacts_inertial[1, i], marker="o", s=500)
        sc.set_facecolor("none")
        sc.set_edgecolor("black")

        arrow_scale=2
        x0 = contacts_inertial[0,i]-contact_vecs[i][0]/arrow_scale
        y0 = contacts_inertial[1,i]-contact_vecs[i][1]/arrow_scale
        plt.arrow(x0,y0, contact_vecs[i][0]/arrow_scale, contact_vecs[i][1]/arrow_scale, head_width=0.05, color="red", length_includes_head=True)
    plt.grid()
    plt.legend()
    plt.title("Scenario")


def scenario_summary_gif():
    from matplotlib.animation import FuncAnimation
    n_frames = 300
    theta = np.linspace(-np.pi/2+.001, np.pi/2-.001, n_frames)
    multi_contact = True
    fig, axs = plt.subplots(1,2)
    def animate(i):
        y_basis = rot(-theta[i])@np.array([0,-1])
        resolution = 10000
        ax = axs[0]
        hole = gen_hole_profile(resolution=resolution)
        peg, peg_corners, _, _ = gen_peg_profile(resolution=resolution)
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
        ax = axs[1]
        ax.clear()

        #contacts_peg = rot(-theta[i])@scenario["contacts_inertial"]
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


        ax.legend()
        ax.set_aspect("equal")
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.grid(alpha=0.2)
        ax.set_xlabel("Peg X")
        ax.set_ylabel("Peg Y")
        ax.set_title("Contacts in Peg Frame")

        fig.tight_layout()
    ani = FuncAnimation(fig, animate, frames=n_frames, interval=0.05, repeat=True)
    plt.show()


def plot_scenario_peg_frame(scenario, theta, fg):
    fig, ax = plt.subplots()
    contacts_peg = scenario["contacts_peg"]
    peg = rot(-theta) @ scenario["peg"]
    corners = rot(-theta) @ scenario["corners"]
    contact_vecs_peg = scenario["contact_vecs_peg"]
    delta = np.mean(corners, axis=1)
    #y_basis = rot(-theta) @ np.array([0, -1])

    ax.plot(peg[0, :] - delta[0], peg[1, :] - delta[1])
    ax.scatter(contacts_peg[0, :], contacts_peg[1, :])
    ax.fill(corners[0, :] - delta[0], corners[1, :] - delta[1], color="blue", alpha=0.2, hatch="/", edgecolor="blue")
    ax.arrow(0, 0, fg[0], fg[1], head_width=.1, color="red", label="Gravity Vector")

    for cid in range(scenario["contacts_inertial"].shape[1]):
        arrow_scale = 2
        x0 = contacts_peg[0, cid] - contact_vecs_peg[cid][0] / arrow_scale
        y0 = contacts_peg[1, cid] - contact_vecs_peg[cid][1] / arrow_scale
        ax.arrow(x0, y0, contact_vecs_peg[cid][0] / arrow_scale, contact_vecs_peg[cid][1] / arrow_scale,
                 head_width=0.05,
                 color="red", length_includes_head=True)

    ax.legend()
    ax.set_aspect("equal")
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.grid(alpha=0.2)
    ax.set_xlabel("Peg X")
    ax.set_ylabel("Peg Y")
    ax.set_title("Contacts in Peg Frame")



if __name__ == "__main__":
    multi_contact = True
    debug_plots = False
    resolution = 10000
    hole = gen_hole_profile(resolution=resolution)
    peg, peg_corners, _, _ = gen_peg_profile(resolution=resolution)
    theta = np.pi/4 # if theta positive, look at left face for second rotation. If negative look at right face.

    scenario = gen_scenario(peg, hole, peg_corners, theta, resolution, multi_contact=True, debug_plots=debug_plots)
    # Finally, get contact points in the peg-fixed reference frame
    plot_scenario(scenario)

    # Rotate peg to vertical
    peg = rot(-theta)@scenario["peg"]
    contacts_peg = rot(-theta)@scenario["contacts_inertial"]
    corners = rot(-theta)@scenario["corners"]

    delta = np.mean(corners, axis=1)

    hole = scenario["hole"]
    plt.figure()
    plt.plot(peg[0,:]-delta[0],  peg[1,:]-delta[1])
    plt.scatter(contacts_peg[0,:]-delta[0], contacts_peg[1,:]-delta[0])
    plt.axis("equal")

    #plt.show()
    scenario_summary_gif()







