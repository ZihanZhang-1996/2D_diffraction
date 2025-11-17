from ase.visualize import view
from ase.build import bulk
from ase.io import read
from ase.io import write
from ase.io import cif
import os

folders=['229']
for folder in folders:
	for files in os.listdir(folder):
		if '.cif' in files:
			atoms1 = read(os.path.join(folder,files))
			save_path=os.path.join(folder,'new.cif')
			if 'a-b' in files:
				atoms1.set_cell([atoms1.cell[1]-atoms1.cell[0], atoms1.cell[1], atoms1.cell[2]])
				atoms1.write(save_path)
			elif 'b-c' in files:
				atoms1.set_cell([atoms1.cell[0], atoms1.cell[1]-atoms1.cell[2], atoms1.cell[2]])
				atoms1.write(save_path)
			elif 'a-c' in files:
				atoms1.set_cell([atoms1.cell[0]-atoms1.cell[2], atoms1.cell[1], atoms1.cell[2]])
				atoms1.write(save_path)
			elif '2a-b' in files:
				atoms1.set_cell([2*atoms1.cell[0]-atoms1.cell[2], atoms1.cell[1], atoms1.cell[2]])
				atoms1.write(save_path)
			else:
				pass
