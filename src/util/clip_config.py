"""
CLIP model word list configurations for semantic exploration.

This module defines sets of words used to create semantic axes in the CLIP embedding space.
These word sets are used to define conceptual dimensions like gender, age, and social status
that can be used to navigate the latent space in a structured way.
"""

# Words representing masculine concepts
masculine = [
    "man",
    "king",
    "prince",
    "husband",
    "father",
]

# Words representing feminine concepts
feminine = [
    "woman",
    "queen",
    "princess",
    "wife",
    "mother",
]

# Words representing adult/mature concepts
old = [
    "man",
    "woman",
    "king",
    "queen",
    "father",
]

# Words representing youth/childhood concepts
young = [
    "boy",
    "girl",
    "prince",
    "princess",
    "son",
]

# Words representing common/everyday people
common = [
    "man",
    "woman",
    "boy",
    "girl",
    "woman",
]

# Words representing royalty/nobility
elite = [
    "king",
    "queen",
    "prince",
    "princess",
    "duchess",
]

# Words representing singular objects/beings
singular = [
    "boy",
    "girl",
    "cat",
    "puppy",
    "computer",
]

# Words representing plural objects/beings
plural = [
    "boys",
    "girls",
    "cats",
    "puppies",
    "computers",
]

# Default example words for the embedding visualization
examples = [
    "king",
    "queen",
    "man",
    "woman",
    "boys",
    "girls",
    "apple",
    "orange",
]

# Names of the semantic axes used in the 3D visualization
axis_names = ["gender", "residual", "age"]

# Word sets used for different semantic axes
# These are used to define the axes of variation in the embedding space
axis_combinations = {
    "age": young + old,            # Axis representing progression from youth to adulthood
    "gender": masculine + feminine, # Axis representing spectrum of gender concepts
    "royalty": common + elite,      # Axis representing social status from common to royal
    "number": singular + plural,    # Axis representing grammatical number (singular/plural)
}

# Mapping of axis labels to indices in the 3D plot
axisMap = {
    "X - Axis": 0,  # First dimension in the 3D space
    "Y - Axis": 1,  # Second dimension in the 3D space
    "Z - Axis": 2,  # Third dimension in the 3D space
}

# Mapping of UI controls to axis selections
# This defines which UI control corresponds to which axis in the visualization
whichAxisMap = {
    "which_axis_1": "X - Axis",  # Control for the X axis
    "which_axis_2": "Z - Axis",  # Control for the Z axis
    "which_axis_3": "Y - Axis",  # Control for the Y axis
    "which_axis_4": "---",       # Unused axis control
    "which_axis_5": "---",       # Unused axis control
    "which_axis_6": "---",       # Unused axis control
}

# Export the public symbols from this module
__all__ = [
    "axisMap",
    "whichAxisMap",
    "axis_names",
    "axis_combinations",
    "examples",
    "masculine",
    "feminine",
    "young",
    "old",
    "common",
    "elite",
    "singular",
    "plural",
]
