
import plotly.graph_objects as go

def clear_axis(figure:go.Figure):
    """ Remove the axis of the figure.
    """

    figure.update_layout(
        scene = dict(
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis = dict(visible=False)
            )
        )
    return figure
    
def set_aspect(figure:go.Figure):
    """ Set the aspect of figure according to data.
    """
    figure.update_layout(
        scene={'aspectmode': 'data'}
    )
    return figure


def set_background_color(figure:go.Figure, color):
    """ Set the background color of a figure.
    """
    figure.update_layout(
        paper_bgcolor=color,
    )
    return figure


def mesh2ply(mesh:go.Mesh3d):
    """ Export a mesh to plyfile.PlyData (requires plyfile package)
    """

    import plyfile
    import numpy as np

    if mesh.vertexcolor is None:
        vertex = np.empty(len(mesh.x), dtype=[
            ('x','f4'),('y','f4'),('z','f4')])
        vertex['x'] = mesh.x
        vertex['y'] = mesh.y
        vertex['z'] = mesh.z
    else:
        vertex = np.empty(len(mesh.x), dtype=[
            ('x','f4'),('y','f4'),('z','f4'),
            ('red','u1'),('green','u1'),('blue','u1')])
        vertex['x'] = mesh.x
        vertex['y'] = mesh.y
        vertex['z'] = mesh.z
        vertex['red'] = mesh.vertexcolor[:,0]
        vertex['green'] = mesh.vertexcolor[:,1]
        vertex['blue'] = mesh.vertexcolor[:,2]

    face = np.empty(len(mesh.i), dtype=[('vertex_indices','i4',(3,))])
    for n in range(len(mesh.i)):
        face['vertex_indices'][n] = (mesh.i[n], mesh.j[n], mesh.k[n])

    return plyfile.PlyData([
        plyfile.PlyElement.describe(vertex,'vertex'),
        plyfile.PlyElement.describe(face,'face'),
    ])
