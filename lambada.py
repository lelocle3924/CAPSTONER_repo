spells = [
    ("niggersaur", 50, 20),
    ("blacksaur", 150, 20),   
    ("retardsaur", 290, 30),
]

efficient_spells = sorted(
    spells,
    key = lambda spell: spell[1]/spell[2],
    reverse = True 
)

for spell in efficient_spells:
    print(f"{spell[0]}: {spell[1]/spell[2]:.2f} CPM")