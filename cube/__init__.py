
import numpy as np
from typing import List

class Cube:

    charges: np.ndarray
    coords: np.ndarray
    origin: np.ndarray
    vecs: np.ndarray
    data: np.ndarray

    def __init__(self, charges:np.ndarray, coords:np.ndarray, origin:np.ndarray, vecs:np.ndarray, data:np.ndarray):
        self.charges = charges
        self.coords = coords
        self.origin = origin
        self.vecs = vecs
        self.data = data
        self.shape = self.data.shape

    def axis_coord(self, axis:int) -> np.ndarray:
        """ Returns a (N x 3) array representing the coordinates of each axis (0, 1, 2) of cube edges.
        """
        return self.origin + self.vecs[axis] * np.arange(self.data.shape[axis])

    def meshgrid(self) -> List[np.ndarray]:
        """ Returns a list of X, Y, Z grids, each with size (Nx x Ny x Nz).
        """
        rawX, rawY, rawZ = np.mgrid[0:self.data.shape[0],0:self.data.shape[1],0:self.data.shape[2]]
        return [
            rawX*self.vecs[0,j] + rawY*self.vecs[1,j] + rawZ*self.vecs[2,j] + self.origin[j]
            for j in range(3)
        ]
        


def read_cube(filename:str) -> Cube:
    """ Read a cube file.
    """
    
    with open(filename, 'r') as f:

        f.readline()
        f.readline()

        ls = f.readline().strip().split()
        natom = int(ls[0])
        origin = np.array(list(map(float, ls[1:])))

        ngrids = []
        vecs = np.zeros((3,3))
        for j in range(3):
            ls = f.readline().strip().split()
            ngrids.append(int(ls[0]))
            vecs[j] = np.array(list(map(float, ls[1:])))

        atoms = []
        charges = []
        coords = []
        for j in range(natom):
            ls = f.readline().strip().split()
            atoms.append(int(ls[0]))
            charges.append(float(ls[1]))
            coords.append(np.array(list(map(float, ls[2:]))))


        data = np.fromstring(f.read(), sep='\n')
        assert len(data) == np.prod(ngrids), "Data length does not match the grid size"

        data = data.reshape(ngrids[0], ngrids[1], ngrids[2])

        return Cube(atoms, np.array(coords), origin, vecs, data)
    

from . import plot
from .plot import plot_volume, plot_isosurface, plot_molecule, gen_bonds


def plot_cube(cube, show_molecule=True, style='mesh', figure=None, mol_kwargs={}, surf_kwargs={}):
    """ Simple wrapper for plotting a cube file.

    cube: `Cube` instance or `str` of cube filename;
    show_molecule: Whether display atoms;
    style: One of `None`/'tranparent'/'shading'/'mesh';
    figure: `None` or exisiting `plotly.graph_object.Figure` instance;
    mol_kwargs, surf_kwargs: Keyword arguments passed to `plot_molecule()` and `plot_isosurface()`.
    """

    if isinstance(cube, str):
        cube = read_cube(cube)
    
    if show_molecule:
        fig = plot_molecule(cube.charges, cube.coords, solid=True, figure=figure, **mol_kwargs)
    else:
        fig = figure

    X, Y, Z = cube.meshgrid()
    return plot_isosurface(X, Y, Z, cube.data, style=style, figure=fig, **surf_kwargs)
    
