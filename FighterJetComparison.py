import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class Jet:
    def __init__(self, name, thrust_kN, MTOW, wingarea_m2, canardarea_m2, cl_alpha_wing, cl_alpha_canard, cbar, cd0, e, ar, cg, xnp, iy, pmarm, wingarm, canardarm):
        self.name = name
        self.thrust_kN = thrust_kN
        self.MTOW = MTOW
        self.weight_kg = 0.75 * MTOW
        self.wingarea_m2 = wingarea_m2
        self.canardarea_m2 = canardarea_m2
        self.cl_alpha_wing = cl_alpha_wing
        self.cl_alpha_canard = cl_alpha_canard
        self.cbar = cbar
        self.cd0 = cd0
        self.e = e
        self.ar = ar
        self.cg = cg
        self.xnp = xnp
        self.iy = iy
        self.pmarm = pmarm
        self.wingarm = wingarm
        self.canardarm = canardarm


jets = [
    Jet('Gripen', 54, 14000, 30, 2.25, 0.08, 0.12, 3.49, 0.02, 0.8, 2.47, 3.8, 4.3, 50000, 1.00, -3.2, 0.7),
    Jet('Rafale', 50, 24000, 45.7, 2.7, 0.09, 0.15, 4.19, 0.02, 0.82, 2.6, 4.5, 5, 80000, 1.15, -4.3, 0.5),
    Jet('Typhoon', 60, 23500, 51.2, 2.7, 0.07, 0.13, 4.67, 0.02, 0.8, 2.34, 4.6, 5.1, 85000, 1.30, -4.8, 0.1),
    Jet('Tejas Mk2', 55, 17500, 41, 3, 0.085, 0.14, 4.32, 0.02, 0.78, 2.5, 3.9, 4.5, 60000, 1.30, -6.8, 1.8)
]


kv = 0.15
alpha_0 = -3
h_ft = np.linspace(0, 35000, 10000)  # Altitude in feet
def ft_to_meters(ft):
    return ft * 0.3048
h = ft_to_meters(h_ft)  # Convert altitude to meters

def isa_density(h):
    t0 = 288.15
    p0 = 101325
    l = 0.0065
    R = 287.05
    g = 9.80665
    t = t0 - l * h
    p = p0*(t/t0)**(g/(l*R))
    density = p / (R * t)
    return density
rho = isa_density(h)
canard_deflection = np.linspace(0,20,10000)
alpha = np.linspace(0, 30, 10000)

try:
    v = float(input("Enter airspeed in m/s: "))
except ValueError:
    print("Invalid input! Using default airspeed of 200 m/s.")
    v = 200


# Use constrained_layout for fig1 and fig2 to prevent text overlap
fig1, ax1 = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
for axes in ax1.flat:
    axes.grid(which='both', linewidth=0.7)
    axes.minorticks_on()

fig2, ax2 = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
for axes in ax2.flat:
    axes.grid(which = 'both', linewidth=0.7)
    axes.minorticks_on()

fig3, ax3 = plt.subplots(2, 2, figsize=(14, 12))
for axes in ax3.flat:
    axes.grid(which='both', linewidth=0.7)
    axes.minorticks_on()

fig4, ax4 = plt.subplots(2, 3, figsize=(14, 12))
for axes in ax4.flat:
    axes.grid(which='both', linewidth=0.7)
    axes.minorticks_on()

fig5, ax5 = plt.subplots(1, 2, figsize=(14, 12))
for axes in ax5.flat:
    axes.grid(which='both', linewidth=0.7)
    axes.minorticks_on()

handles = []
labels = []


for jet in jets:
    cl_canard = jet.cl_alpha_canard * (np.radians(canard_deflection) + np.radians(alpha) - alpha_0)*(1-np.exp(-(alpha + canard_deflection)/30)**2)

    if jet.name == "Rafale" or jet.name == "Tejas Mk2":
        cl_wing = jet.cl_alpha_wing * np.radians(alpha - alpha_0)*(1-np.exp(-(alpha/30)**2))*(1 + kv)
    else:
        cl_wing = jet.cl_alpha_wing * np.radians(alpha)

    l_wing = 0.5*rho*v**2*jet.wingarea_m2*cl_wing
    l_canard = 0.5*rho*v**2*jet.canardarea_m2*cl_canard
    l_total = l_wing + l_canard
    cl = (l_wing + l_canard)/(0.5*rho*v**2*(jet.wingarea_m2))

    cm = (l_canard*jet.canardarm - l_wing*jet.wingarm)/(0.5*rho*v**2*(jet.wingarea_m2))
    m = 0.5*rho*v**2*jet.wingarea_m2*jet.cbar*cm


    h0, = ax1[0,0].plot(alpha, cl, label=f'{jet.name}')
    ax1[0,0].set_title('Total Lift Coefficient v/s Angle of Attack')
    ax1[0,0].set_xlabel('Angle of Attack')
    ax1[0,0].set_ylabel('Total Lift Coefficient')
    handles.append(h0)
    labels.append(f'{jet.name}')

    ax1[0,1].plot(alpha, cl_wing)
    ax1[0,1].set_title('Wing Lift Coefficient v/s Angle of Attack') 
    ax1[0,1].set_xlabel('Angle of Attack')
    ax1[0,1].set_ylabel('Wing Lift Coefficient')

    ax1[1,0].plot(alpha, cl_canard)
    ax1[1,0].set_title('Canard Lift Coefficient v/s Angle of Attack')
    ax1[1,0].set_xlabel('Angle of Attack')
    ax1[1,0].set_ylabel('Canard Lift Coefficient')

    ax1[1,1].plot(canard_deflection, cl)
    ax1[1,1].set_title('Total Lift Coefficient v/s Deflection Angle')
    ax1[1,1].set_xlabel('Deflection Angle')
    ax1[1,1].set_ylabel('Total Lift Coefficient')

    ax1[2,0].plot(canard_deflection, cl_wing)
    ax1[2,0].set_title('Wing Lift Coefficient v/s Deflection Angle')
    ax1[2,0].set_xlabel('Deflection Angle')
    ax1[2,0].set_ylabel('Wing Lift Coefficient')

    ax1[2,1].plot(canard_deflection, cl_canard)
    ax1[2,1].set_title('Canard Lift Coefficient v/s Deflection Angle')
    ax1[2,1].set_xlabel('Deflection Angle') 
    ax1[2,1].set_ylabel('Canard Lift Coefficient')

    h1, = ax2[0,0].plot(h, l_total)
    ax2[0,0].set_title('Total Lift v/s Altitude')
    ax2[0,0].set_xlabel('Altitude')
    ax2[0,0].set_ylabel('Total Lift')

    ax2[0,1].plot(l_canard, l_wing)
    ax2[0,1].set_title('Canard Lift v/s Wing Lift')
    ax2[0,1].set_xlabel('Canard Lift')
    ax2[0,1].set_ylabel('Wing Lift')

    ax2[1,0].plot(l_canard, l_total)
    ax2[1,0].set_title('Canard Lift v/s Total Lift')
    ax2[1,0].set_xlabel('Canard Lift')
    ax2[1,0].set_ylabel('Total Lift')

    ax2[1,1].plot(l_wing, l_total)
    ax2[1,1].set_title('Wing Lift v/s Total Lift')
    ax2[1,1].set_xlabel('Wing Lift')
    ax2[1,1].set_ylabel('Total Lift')

    ax2[2,0].plot(canard_deflection, l_total)
    ax2[2,0].set_title('Total Lift v/s Deflection Angle')
    ax2[2,0].set_xlabel('Deflection Angle')
    ax2[2,0].set_ylabel('Total Lift')

    ax2[2,1].plot(canard_deflection, l_wing)
    ax2[2,1].set_title('Wing Lift v/s Deflection Angle')
    ax2[2,1].set_xlabel('Deflection Angle')
    ax2[2,1].set_ylabel('Wing Lift')

    h3, = ax4[0,0].plot(h,cm)
    ax4[0,0].set_title('Height v/s Pitching Moment Coefficient')
    ax4[0,0].set_xlabel('Height')
    ax4[0,0].set_ylabel('Pitching Moment Coefficient')
    
    ax4[0,1].plot(h,m)
    ax4[0,1].set_title('Height v/s Pitching Moment')
    ax4[0,1].set_xlabel('Height')
    ax4[0,1].set_ylabel('Pitching Moment')

    ax4[0,2].plot(alpha, cm)
    ax4[0,2].set_title('Angle of Attack v/s Pitching Moment Coefficient')
    ax4[0,2].set_xlabel('Angle of Attack')
    ax4[0,2].set_ylabel('Pitching Moment Coefficient')
    
    ax4[1,0].plot(canard_deflection, cm)
    ax4[1,0].set_title('Canard Deflection Angle v/s Pitching Moment Coefficient')
    ax4[1,0].set_xlabel('Canard Deflection Angle')
    ax4[1,0].set_ylabel('Pitching Moment Coefficient')

    ax4[1,1].plot(l_total, cm)
    ax4[1,1].set_title('Total Lift v/s Pitching Moment Coefficient')
    ax4[1,1].set_ylabel('Pitching Moment Coefficient')
    ax4[1,1].set_xlabel('Total Lift')
    
    ax4[1,2].plot(cl,cm)
    ax4[1,2].set_title('Coefficient of Lift v/s Pitching Moment Coefficient')
    ax4[1,2].set_xlabel('Coefficient of Lift')
    ax4[1,2].set_ylabel('Pitching Moment Coefficient')


fig1.legend(handles, labels, loc='upper center', ncol=len(jets), bbox_to_anchor=(0.5, 1), fontsize=8, frameon=True)
fig2.legend(handles, labels, loc='upper center', ncol = len(jets), bbox_to_anchor=(0.5, 1), fontsize=8, frameon=True)
fig3.legend(handles, labels, loc='upper center', ncol=len(jets), bbox_to_anchor=(0.55, 1), fontsize=8, frameon=True)
fig4.legend(handles, labels, loc='upper center', ncol=len(jets), bbox_to_anchor=(0.5, 1), fontsize=8, frameon=True)
fig5.legend(handles, labels, loc='upper center', ncol=len(jets), bbox_to_anchor=(0.5, 1), fontsize=8, frameon=True)

plt.tight_layout(rect=[0,0,1,0.93], pad=4.0, h_pad=3.5, w_pad=3.5)
#plt.show()

try:
    user_height = float(input("Enter height in ft: "))
    if 0 < user_height < 35000:
        user_height_m = ft_to_meters(user_height)
    else:
        print("Height out of range. Please enter a value between 0 and 35000 ft. Using default 10k ft.")
        user_height_m = ft_to_meters(10000)
except ValueError:
    print("Invalid input! Using default height of 10k ft.")
    user_height_m = ft_to_meters(10000)

rho2 = isa_density(user_height_m)

vel_range = np.linspace(100, 200, 10000)

for jet in jets:
    cl_canard = jet.cl_alpha_canard * (np.radians(canard_deflection) + np.radians(alpha) - alpha_0)*(1-np.exp(-(alpha + canard_deflection)/30)**2)
    if jet.name == "Rafale" or jet.name == "Tejas Mk2":
        cl_wing = jet.cl_alpha_wing * np.radians(alpha - alpha_0)*(1-np.exp(-(alpha/30)**2))*(1 + kv)
    else:
        cl_wing = jet.cl_alpha_wing * np.radians(alpha)
    for v in vel_range:
        l_wing = 0.5*rho2*v**2*jet.wingarea_m2*cl_wing
        l_canard = 0.5*rho2*v**2*jet.canardarea_m2*cl_canard
        l_total = l_wing + l_canard
        cl = (l_wing + l_canard)/(0.5*rho2*v**2*(jet.wingarea_m2))

        cm = (l_canard*jet.canardarm - l_wing*jet.wingarm)/(0.5*rho2*v**2*(jet.wingarea_m2))
        m = 0.5*rho2*v**2*jet.wingarea_m2*jet.cbar*cm
        sm = (((jet.xnp - jet.cg)*100)/jet.cbar)

    h2, = ax3[0,0].plot(vel_range, cl, label=f'{jet.name} at {user_height:.0f} ft')
    ax3[0,0].set_title('Total Lift Coefficient v/s Airspeed')
    ax3[0,0].set_xlabel('Velocity')
    ax3[0,0].set_ylabel('Total Lift Coefficient')

    ax3[0,1].plot(vel_range, l_total)
    ax3[0,1].set_title('Total Lift v/s Airspeed')
    ax3[0,1].set_xlabel('Velocity')
    ax3[0,1].set_ylabel('Total Lift')
    
    ax3[1,0].plot(vel_range, l_canard)
    ax3[1,0].set_title('Canard Lift v/s Airspeed')
    ax3[1,0].set_xlabel('Velocity')
    ax3[1,0].set_ylabel('Canard Lift')

    ax3[1,1].plot(vel_range, l_wing)
    ax3[1,1].set_title('Wing Lift v/s Airspeed')
    ax3[1,1].set_xlabel('Velocity')
    ax3[1,1].set_ylabel('Wing Lift')

    ax5[0].plot(vel_range, cm)
    ax5[0].set_title('Velocity v/s Pitching Moment Coefficient')
    ax5[0].set_xlabel('Velocity')
    ax5[0].set_ylabel('Pitching Moment Coefficient')

    ax5[1].plot(vel_range, m)
    ax5[1].set_title('Velocity v/s Pitching Moment')
    ax5[1].set_xlabel('Velocity')
    ax5[1].set_ylabel('Pitching Moment')


plt.show()