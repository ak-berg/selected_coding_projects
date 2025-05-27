# Python Skript für die Erstellung eines zirkularen Balkendiagrammes.
# Die Datensätze in 'dataset' stammen dabei aus Power BI.

# Der folgende Code zum Erstellen eines Datenrahmens und zum Entfernen doppelter Zeilen wird immer ausgeführt und dient als Präambel für Ihr Skript:

# dataset = pandas.DataFrame(Kennzahl, Kennzahl_Relativ, Bereich, Anzahl_Kennzahlen, Alle_Bereiche)
# dataset = dataset.drop_duplicates()

# Fügen oder geben Sie hier Ihren Skriptcode ein:

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# Build a dataset
df = pd.DataFrame({
    "name": dataset["Kennzahl"],
    "value": 100*dataset["Kennzahl_Relativ"],
    "group": dataset["Bereich"],
    "kennzahlen": dataset["Anzahl_Kennzahlen"],
    "allGroups": dataset["Alle_Bereiche"]
})

def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if 0 < angle < np.pi:
        alignment = "right"
        rotation = rotation + 180
    elif np.pi < angle < 2 * np.pi:
        alignment = "left"
    else:
        alignment = "center"
    return rotation, alignment

def add_labels(angles, values, labels, offset, ax, font_size):

    # This is the space between the end of the bar and the label
    padding = 20

    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle

        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            backgroundcolor="white",
            x=angle,
            y=value + padding,
            fontsize=font_size,
            s=label,
            ha=alignment,
            va="center",
            rotation=0*rotation,
            rotation_mode="anchor"
        )

ANGLES = np.linspace(0, 2 * np.pi, len(df), endpoint=False)
VALUES = df["value"].values
LABELS = df["name"].values
MAX_KENNZAHLEN = df["kennzahlen"].values
ALL_GROUPS = np.array([x.split(',') for x in np.unique(df["allGroups"].values)])[0]

# Determine the width of each bar.
# The circumference is '2 * pi', so we divide that total width over the number of bars.
WIDTH = 2 * np.pi / len(VALUES)

# Determines where to place the first bar.
# By default, matplotlib starts at 0 (the first bar is horizontal)
# but here we say we want to start at pi/2 (90 deg)
OFFSET = np.pi / 2

# Grab the group values
GROUP = df["group"].values

#bereiche = ["Finanzen", "Personal", "Lehre"]
groups_selected = np.isin(ALL_GROUPS, np.unique(GROUP))

# Add one empty bar to the end of each group
PAD = 1
if len(np.unique(GROUP)) == 1:
    PAD = 0
ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)

# Obtain size of each group
GROUPS_SIZE = [len(i[1]) for i in df.groupby("group",sort=False)]

# Obtaining the right indexes is now a little more complicated
offset = 0
IDXS = []
for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + size + PAD))
    offset += size + PAD

# Defining layout and plot-type
fig, ax = plt.subplots(figsize=(66, 30), subplot_kw={"projection": "polar"})

ax.set_theta_offset(OFFSET)
ax.set_ylim(-100, 1.025*df["value"].max())
ax.set_frame_on(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Get maximal value of Kennzahl and set scaling parameters
d_off = VALUES.max()/15
d = np.stack(((100+d_off)*np.ones(len(VALUES)),VALUES+d_off),axis=1)
dmax = np.max(d, axis=1)
if VALUES.max() < 100:
    dmax = (VALUES.max()+d_off)*np.ones(len(VALUES))
diff_max = VALUES.max()-100

# Scaling of the labels and grid size (20% step circles and space between neighboring bars)
label_size = 1*9*np.sqrt(2*MAX_KENNZAHLEN.max()-len(LABELS)) - (2*diff_max*100 + diff_max**2)/5000
grid_size = label_size/6

if len(np.unique(GROUP)) > 1:
    #Set radial axis
    radial_ticks = [20, 40, 60, 80, 100]
    ax.set_rlabel_position(2)
    plt.yticks(radial_ticks, color='black', size=6*label_size/7)
    ax.set_yticks(radial_ticks)
    # Set format of radial axis to percent
    yticks = mtick.FuncFormatter('{0}%'.format)
    ax.yaxis.set_major_formatter(yticks)

# Define colorset
Cset = np.array([(204,204,204),(232,209,102),(112,187,255),(113,123,197),(240,167,135),(166,102,176),(236,143,202)])*(1/255)
Cset_dynam = [Cset[i] for i in groups_selected.nonzero()[0]]
# Use different colors for each group!
GROUPS_SIZE = [len(i[1]) for i in df.groupby("group",sort=False)]
COLORS = [Cset_dynam[i] for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

# And finally add the bars.
# Note again the `ANGLES[IDXS]` to drop some angles that leave the space between bars.
ax.bar(
    ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS,
    edgecolor="white", linewidth=grid_size
)


# Add labels to each bar in the chart
LABELS_VALUES = [LABELS[i] + "\n" + str(int(np.round(VALUES[i]))) + "%" for i in range(len(VALUES))]
add_labels(ANGLES[IDXS], dmax, LABELS_VALUES, OFFSET, ax, label_size)


# Extra customization below here --------------------

# This iterates over the sizes of the groups adding reference
# lines and annotations.

#inner_labels = ["Finanzen", "Personal", "Lehre"]
inner_labels_dynam = [ALL_GROUPS[i] for i in groups_selected.nonzero()[0]]

offset = 0
for group, size in zip(inner_labels_dynam, GROUPS_SIZE):
    # Add line below bars
    x1 = np.zeros(50)
    if len(np.unique(GROUP)) > 1:
        x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
        ax.plot(x1, [-5] * 50, color="#333333", lw=4)
    # Add text to indicate group
        textbox = ax.text(0, 0, group, color="#333333", fontsize=7*label_size/6, fontweight="bold", ha="center", va="center")
        ax.text(
            np.mean(x1), -(textbox.get_window_extent().width*abs(np.sin(np.mean(x1))) + textbox.get_window_extent().height*abs(np.cos(np.mean(x1))) + 150)/10, group, color="#333333", fontsize=7*label_size/6,
            fontweight="bold", ha="center", va="center"
        )
        textbox.remove()
    else:
        ax.text(
            np.mean(x1), -100, group, color="#333333", fontsize=7*label_size/6,
            fontweight="bold", ha="center", va="center"
        )

    offset += size + PAD

# Reference lines at steps of 20% until maximal Kennzahl-value
theta = np.arange(0, 2 * np.pi, 0.001)
for i in range(int(VALUES.max()/20)):
    ax.plot(theta, [20*(i+1)] * len(theta), color="white", lw = grid_size)

# U_15/ExU level (100 %)
num_dash = 100
dash_points = 10
dl = 2*np.pi/num_dash
dr = [100] * dash_points
if VALUES.max() >= 100:
    for i in range(int(num_dash/2)):
        ax.plot(np.linspace(2*i*dl, (2*i+1)*dl, dash_points),dr, color="grey", lw = grid_size)

# Show plot
plt.tight_layout(pad=10)
plt.show()
