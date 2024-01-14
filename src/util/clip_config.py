masculine = [
    "man",
    "king",
    "prince",
    "husband",
    "father",
]

feminine = [
    "woman",
    "queen",
    "princess",
    "wife",
    "mother",
]

young = [
    "man",
    "woman",
    "king",
    "queen",
    "father",
]

old = [
    "boy",
    "girl",
    "prince",
    "princess",
    "son",
]

common = [
    "man",
    "woman",
    "boy",
    "girl",
    "woman",
]

elite = [
    "king",
    "queen",
    "prince",
    "princess",
    "duchess",
]

singular = [
    "boy",
    "girl",
    "cat",
    "puppy",
    "computer",
]

plural = [
    "boys",
    "girls",
    "cats",
    "puppies",
    "computers",
]

examples = [
    "king", 
    "queen", 
    "man",
    "woman",
    "boy",
    "child",
    "apple",
    "orange",
]

axis_names = [
    "gender", 
    "residual", 
    "age"
]

axis_combinations = {
    "age"       :   young + old,
    "gender"    :   masculine + feminine,
    "royalty"   :   common + elite,
    "number"    :   singular + plural,
}

axisMap = { 
    "X - Axis": 0,
    "Y - Axis": 1,
    "Z - Axis": 2,
}
