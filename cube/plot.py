
from plotly import graph_objects as go
import plotly.colors
import numpy as np
from typing import Optional, Union
from . import constants

def plot_volume(X:np.ndarray, Y:np.ndarray, Z:np.ndarray, value:np.ndarray, vmin:float=0.02, vmax:float=1.0, 
                   relative_scale:bool=True, opacity:float=0.9, auto_opacity:bool=True, 
                   nslice:int=20, colormap=None, colorbar:bool=True,
                   label='', figure:Optional[go.Figure]=None, **kwargs) -> go.Figure:
    """ Plot a volumetric graph by repeatedly drawing isosurfaces.

    X, Y, Z, value: (Nx x Ny x Nz) float array for the 3D data;
    vmin, vmax: Controls the range to draw the density. Sets "isomin" and "isomax" in `Volume()`.
    relative_scale: Whether scale vmin/vmax according to data distribution;
    opacity: The opacity amount;
    auto_opacity: Set opacity based on data: set 0 as most transparent. Sets "opacityscale" in `Volume()`;
    nslice: Number of slices of isosurface plots to enumlate the volume plot;
    colormap: The palette. Sets "colorscale" in `Volume()`;
    colorbar: Whether to draw the colorbar. Sets "showscale" in `Volume()`;
    label: Label of the data. Sets "name" and "showlegend" in `Volume()`.
    figure: If provided, draws on the exisiting figure; Otherwise creates a new figure.

    Additional kwargs will be passed to `plotly.graph_objects.Volume()` and will override existing arguments.
    """
    
    dataabs = np.abs(value)
    dataave = np.average(dataabs[dataabs > 1e-6])
    if np.any(value < -1e-6):
        datanegave = np.average(np.abs(value)[value < -1e-6])
        has_negative = datanegave > dataave / 10 or not np.isnan(datanegave)
    else:
        has_negative = False

    if relative_scale:
        if has_negative:
            datapos = value[value > 1e-6]
            dataneg = value[value < -1e-6]
            dataavepos = np.mean(datapos)
            dataaveneg = np.mean(dataneg)

            refedge = max(dataavepos + 8*min(np.std(datapos),dataavepos,0.1*np.max(datapos)), 
                          -dataaveneg + 8*min(np.std(dataneg),-dataaveneg,-0.1*np.min(dataneg)))
            vmin = -refedge * vmax
            vmax = refedge * vmax
        else:
            refedge = dataave + 8*min(np.std(dataabs), dataave, 0.1*np.max(value))
            vmin = refedge * vmin
            vmax = refedge * vmax

    if auto_opacity:
        if has_negative: # Has negative
            opacityscale = [[0,opacity],[0.4,opacity*0.8],[0.5,0],[0.6,opacity*0.8],[1,opacity]]
        else:
            opacityscale = [[0,0],[0.4,opacity*0.8],[1,opacity]]
    else:
        opacityscale = 'uniform' if not has_negative else 'extremes'

    if colormap is None:
        colormap = 'Picnic' if has_negative else 'Plasma'

    volume_kwargs = dict(
        isomin=vmin, isomax=vmax, opacity=opacity, surface_count=nslice, 
        colorscale=colormap, autocolorscale=colormap is None, showscale=colorbar,
        opacityscale=opacityscale
    )

    if label:
        volume_kwargs['name'] = label
        volume_kwargs['showlegend'] = True
    
    volume_kwargs.update(kwargs)

    frame_data = [go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=value.flatten(),
        **volume_kwargs
    )]

    if figure is None:
        figure = go.Figure(data=frame_data)
    else:
        figure.add_traces(frame_data)

    return figure



def plot_isosurface(X:np.ndarray, Y:np.ndarray, Z:np.ndarray, value:np.ndarray, iso:float=0.4, relative_scale:bool=True,
                    draw_negative:Optional[bool]=None, style:Optional[str]=None, color:str='DarkOrchid', color2:Optional[str]=None, 
                    opacity:float=1.0, smooth_grid:bool=False, label:Optional[str]=None, figure:Optional[go.Figure]=None, **kwargs):
    """ Plot an isosurface at value==iso.

    X, Y, Z, value: (Nx x Ny x Nz) float array for the 3D data;
    iso: The value to compute the isosurface. If `relative_scale` is set, will draw according to data distribution;
    relative_scale: Whether draw isosurface according to data distribution;
    draw_negative: If data has both positive and negative part, will draw two surfaces instead of one. If `None`, detects
        from data distribution;
    style: None/'transparent'/'shading'/'liquid'/'bubble'/'mesh'. A shorthand of controlling multiple parameters of the style;
    color, color2: Colors for the positive and negative part;
    opacity: Controls the opacity;
    smooth_grid: Generate smoother surfaces by allowing to plot at subgrids. Requires scikit-image; Not compatible with style = mesh;
    label: Label of the data. If not `None`, sets the "name=label" and "showlegend=True" in `Isosurface()`;
    figure: If provided, draws on the exisiting figure; Otherwise creates a new figure.

    Additional kwargs will be passed to `plotly.graph_objects.Isosurface()` (or `Mesh3d()`, if smooth_grid = True)
        and will override existing arguments.
    """

    if draw_negative is None:
        dataabs = np.abs(value)
        dataave = np.average(dataabs[dataabs > 1e-6])
        if np.any(value < -1e-6):
            datanegave = np.average(np.abs(value)[value < -1e-6])
            has_negative = datanegave > dataave / 10 or not np.isnan(datanegave)
        else:
            has_negative = False
    else:
        has_negative = draw_negative

    if relative_scale:
        datapos = value[value > 1e-6]
        dataavepos = np.mean(datapos)
        if has_negative:
            dataneg = value[value < -1e-6]
            dataaveneg = np.mean(dataneg)

            refedge = max(dataavepos + 8*min(np.std(datapos),dataavepos,0.1*np.max(datapos)), 
                          -dataaveneg + 8*min(np.std(dataneg),-dataaveneg,-0.1*np.min(dataneg)))
            isomin = -refedge * iso
            isomax = refedge * iso
        else:
            refedge = dataavepos + 8*min(np.std(datapos), dataavepos, 0.1*np.max(value))
            isomin = refedge * iso
            isomax = refedge * iso + 1e-5
    else:
        if has_negative:
            isomin, isomax = -iso, iso
        else:
            isomin, isomax = iso, iso + 1e-5

    if not color2:
        if style == 'mesh':
            color2 = 'Gold'
        else:
            color2 = '#ffeb00' # TODO translate based on color1

    if has_negative:
        surfcount = 2
        colorscale = [(0,color2),(1,color)]
    else:
        surfcount = 1
        colorscale = [(0,color),(1,color)]

    isosurface_kwargs = dict(
        isomin=isomin, isomax=isomax, opacity=opacity, showscale=False, 
        surface=dict(show=True, count=surfcount), colorscale=colorscale,
        caps=go.isosurface.Caps(x=dict(show=False), y=dict(show=False),z=dict(show=False)),
    )

    if style == 'transparent':
        isosurface_kwargs.update(opacity=0.4)
    elif style == 'shading':
        isosurface_kwargs.update(
            opacity=opacity,
            lighting=dict(ambient=0.5, specular=1, roughness=0.1, fresnel=0.4),
            lightposition=dict(x=X.flat[-1],y=Y.flat[-1],z=Z.flat[-1]),
        )
    elif style == 'liquid':
        isosurface_kwargs.update(
            opacity=0.4,
            lighting=dict(ambient=0.5, specular=1, roughness=0.1, fresnel=0.4),
            lightposition=dict(x=X.flat[-1],y=Y.flat[-1],z=Z.flat[-1]),
        )
    elif style == 'bubble':
        isosurface_kwargs.update(
            opacity=0.2,
            lighting=dict(ambient=0.5, specular=1, roughness=0.1, fresnel=0.2),
            lightposition=dict(x=X.flat[-1],y=Y.flat[-1],z=Z.flat[-1]),
        )
    elif style == 'mesh':
        isosurface_kwargs.update(opacity=opacity, flatshading=True)
        isosurface_kwargs['surface'].update(fill=0.5)
    else:
        isosurface_kwargs['opacity'] = opacity

    if label:
        isosurface_kwargs['name'] = label
        isosurface_kwargs['showlegend'] = True
    
    isosurface_kwargs.update(kwargs)

    if not smooth_grid or style == 'mesh':
        frame_data = go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=value.flatten(),
            **isosurface_kwargs
        )
    else:
        frame_data = _plot_isosurface_ex(X, Y, Z, value, has_negative, **isosurface_kwargs)

    if figure is None:
        figure = go.Figure(data=frame_data)
    else:
        figure.add_trace(frame_data)

    return figure


def _plot_isosurface_ex(X, Y, Z, value, has_negative, **kwargs):

    from skimage import measure

    if not has_negative:
        verts, faces = measure.marching_cubes(value, kwargs['isomin'])[:2]
        

        del kwargs['isomin'], kwargs['isomax'], kwargs['surface'], kwargs['caps']


        return go.Mesh3d(
            x=np.interp(verts[:,0], np.arange(X.shape[0]), X[:,0,0]),
            y=np.interp(verts[:,1], np.arange(Y.shape[1]), Y[0,:,0]),
            z=np.interp(verts[:,2], np.arange(Z.shape[2]), Z[0,0,:]),
            i=faces[:,0],
            j=faces[:,1],
            k=faces[:,2],
            cmin=0,
            cmax=1,
            intensity=np.zeros(len(verts)),
            **kwargs
        )
    
    else:
        
        verts1, faces1 = measure.marching_cubes(value, kwargs['isomin'])[:2]
        verts2, faces2 = measure.marching_cubes(value, kwargs['isomax'])[:2]
        verts = np.concatenate((verts1, verts2), axis=0)
        faces = np.concatenate((faces1, faces2+len(verts1)), axis=0)

        del kwargs['isomin'], kwargs['isomax'], kwargs['surface'], kwargs['caps']

        return go.Mesh3d(
            x=np.interp(verts[:,0], np.arange(X.shape[0]), X[:,0,0]),
            y=np.interp(verts[:,1], np.arange(Y.shape[1]), Y[0,:,0]),
            z=np.interp(verts[:,2], np.arange(Z.shape[2]), Z[0,0,:]),
            i=faces[:,0],
            j=faces[:,1],
            k=faces[:,2],
            cmin=0,
            cmax=1,
            intensity=np.concatenate((np.zeros(len(verts1)), np.ones(len(verts2)))),
            **kwargs
        )


def plot_molecule(atoms:list, coords:Optional[np.ndarray]=None, bonds=None, style:Optional[str]=None, marker_size:int=25, line_width:int=8,
                    edge_width:int=1, edge_color='black', colormap:str='fixed', solid:bool=False, naming_mode:str='simpleid', 
                    name_hydrogen:bool=True, show_element_legend:bool=True, label:Optional[str]=None, conn_cutoff:float=2.0, max_neighbor:int=6, 
                    figure:Optional[go.Figure]=None, marker_kwargs:dict={}, scatter_kwargs:dict={}, line_kwargs:dict={}):
    """ Plot a geometry using plotly.
    atoms: list of atom names;
    coords: N x 3 array of Cartesian coordinates;
    bonds: None, N x 2 integer array of connectivities;
    style: None/'ballstick'/'ballstick2'/'stick'/'ball'. Controls the style of the graph.
        - None: Uses simple 2D objects. Easier for displaying labels and controling the visibility;
        - Others: Will plot 3D model in different styles. For ballstick2/stick/ball, the marker_size and line_width will be overriden;
    marker_size: Referenced marker size in pixels (will be scaled in 3D mode);
    line_width: Width of bond in pixels (will be scaled in 3D mode);
    edge_width: Width of marker edges;
    edge_color: Color of marker edges;
    colormap: "fixed" or one of the palette name.
        - palette (default): Using matplotlib palette to assign colors;
        - fixed (TBI): Using fixed color for each element;
    solid: Deprecated. Equivalent to style='ballstick';
    naming_mode: none/element/simpleid/fullid
        - none: Do not display name;
        - element: Display element names;
        - simpleid (default): Display element name + relative index within the same elements;
        - fullid: Display element name + full index;
    name_hydrogen: Whether display names for hydrogen.
    show_element_legend: Show legend for each element;
    label: Label of the molecule, or None. Turns on the legend;
    conn_cutoff, max_neighbor: Arguments for generating chemical bonds;
    figure: plotly.graph_objects.Figure; 
    scatter_kwargs, marker_kwargs, line_kwargs: Additional argument passed to scatter3D for points, the marker kwarg for 
        scatter3D for points, and the line kwarg for scatter3D for bonds.

    Additional kwargs will be passed to gen_plot_args().
    
    Returns the Figure instance.
    """
    assert len(atoms) == len(coords)

    if isinstance(atoms[0], str):
        names = atoms
        charges = [constants.ElementCharge[a] for a in atoms]
    else:
        charges = atoms
        names = [constants.ElementName[c-1] for c in charges]

    unique_names = set((n for n in names))
    elem_ids = dict(zip(unique_names, range(len(unique_names)))) # Name -> id

    # Build colors
    if colormap == 'fixed':
        colors = [constants.JMolColors[c-1] for c in charges]
    elif isinstance(colormap, str):
        from plotly.colors import qualitative
        palette = qualitative.__dict__[colormap]
        colors = [palette[elem_ids[n]] for n in names]
    elif isinstance(colormap, list):
        colors = [colormap[elem_ids[n]] for n in names]
    else:
        raise ValueError(colormap)
    
    colors = np.array(colors)

    # Build sizes
    periods = [np.searchsorted(constants.Periods, c) for c in charges]
    sizes = np.array([min(4, p+1) for p in periods])

    # Build texts
    if naming_mode == 'none':
        texts = np.array([''] * len(atoms))
    elif naming_mode == 'element':
        texts = np.array(names)
    elif naming_mode == 'simpleid':
        atom_counter = {}
        dtext = []
        for n in names:
            if n not in atom_counter:
                atom_counter[n] = 1
            dtext.append(n + str(atom_counter[n]) )
            atom_counter[n] += 1
        texts = np.array(dtext)
    elif naming_mode == 'fullid':
        texts = np.array([n+str(j) for j,n in enumerate(names)])
    else:
        raise ValueError(naming_mode)
        
    if not name_hydrogen and naming_mode != 'none':
        for j in range(len(texts)):
            if names[j] == 'H':
                texts[j] = ''

    # Bonds
    if bonds is None:
        bonds = gen_bonds(charges, coords, conn_cutoff=conn_cutoff, max_neighbor=max_neighbor)

    if solid and not style:
        style = 'ballstick'

    if style:
        if style == 'ballstick2':
            line_width = 4
            marker_size = 20
        elif style == 'ballstick3':
            line_width = 20
            marker_size = 30
        elif style == 'ball':
            line_width = 0
            marker_size = 80
        elif style == 'stick':
            marker_size = line_width

        marker_kwargs1 = dict(
            size=sizes*marker_size/80 if style != 'stick' else np.ones(len(sizes))*marker_size/80, color=colors
        )

        if label:
            scatter_kwargs1 = dict(name=label, showlegend=True)
        else:
            scatter_kwargs1 = {}
        
        light_pos = np.max(coords, axis=0) + 1.0
        light_pos /= np.linalg.norm(light_pos)
        scatter_kwargs1.update(marker=marker_kwargs1,
                               lighting=dict(ambient=0.3, specular=0.5, roughness=0.1, fresnel=0.2),
                               lightposition=dict(
                                    x=light_pos[0],
                                    y=light_pos[1],
                                    z=light_pos[2])
                              )
        scatter_kwargs1.update(scatter_kwargs)

        frame_data = _plot_molecule_ballstick(coords, bonds, line_width/80, **scatter_kwargs1)
        # TODO show text
        # frame_data.append(
        #     go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], text=texts,
        #         mode='markers+text',
        #         marker=marker_kwargs,)
        # )

    elif not show_element_legend:
        marker_kwargs1 = dict(
            size=sizes*marker_size, opacity=1, color=colors,
            line=dict(color=edge_color, width=edge_width)
        )
        marker_kwargs1.update(marker_kwargs)
        if label:
            scatter_kwargs1 = dict(
                name=label, showlegend=True,
            )
        else:
            scatter_kwargs1 = {}
        scatter_kwargs1.update(scatter_kwargs)

        frame_data = [
            go.Scatter3d( 
                x=coords[:,0], y=coords[:,1], z=coords[:,2], text=texts,
                mode='markers+text',
                marker=marker_kwargs1,
                **scatter_kwargs1,
            )
        ]
    else:
        frame_data = []
        group = np.array([elem_ids[n] for n in names])
        for j, name in enumerate(list(unique_names)): 
            f = group == j
            marker_kwargs1 = dict(size=sizes[f]*marker_size, opacity=1, color=colors[f], 
                line=dict(color=edge_color, width=edge_width), **marker_kwargs
            )
            marker_kwargs1.update(marker_kwargs)

            scatter_kwargs1 = dict(
                name=(label + ':' + name) if label else name, showlegend=True,
            )
            scatter_kwargs1.update(scatter_kwargs)

            frame_data.append(
                go.Scatter3d(x=coords[f,0], y=coords[f,1], z=coords[f,2], text=texts[f],
                    mode='markers+text', 
                    marker=marker_kwargs1,
                    **scatter_kwargs1,
                )
            )

    if not style and line_width > 0:
        for p, q in bonds:
            frame_data.append(go.Scatter3d(x=coords[[p,q],0], y=coords[[p,q],1], z=coords[[p,q],2], mode='lines', showlegend=False,
                line=dict(color='black', width=line_width, **line_kwargs)))
    
    if figure is None:
        figure = go.Figure(data=frame_data)
    else:
        figure.add_traces(frame_data)

    return figure


def _gen_mesh_sphere(center, radius, n_theta:int=20, n_phi:int=40):
    
    Theta, Phi = np.meshgrid(np.arange(1,n_theta+1)*np.pi/(n_theta+1), np.arange(n_phi)*np.pi*2/n_phi, indexing='ij')
    X, Y, Z = np.sin(Theta)*np.cos(Phi), np.sin(Theta)*np.sin(Phi), np.cos(Theta)
    verts = np.concatenate((
        np.array([X.flatten(), Y.flatten(), Z.flatten()]).T,
        np.array([[0,0,1]]), np.array([[0,0,-1]])
    ), axis=0) * radius + center

    # We have total (n_theta*n_phi) points. Except for the last row (the "body"), 
    # we connect a point with its bottom (+n_phi) and its right (+1). The rightmost connect to the leftmost -- so we need %
    # All tops connects to the "top" point and all bottom connects to the "bottom" point
    Nbody = np.arange(0, (n_theta-1)*n_phi, dtype=int)
    Nbody_base = (Nbody//n_phi)*n_phi
    Ntop = np.arange(n_phi, dtype=int)
    Nbottom = np.arange((n_theta-1)*n_phi, n_theta*n_phi, dtype=int)

    faces = np.concatenate((
        np.array([Nbody, Nbody+n_phi, Nbody_base + (Nbody+1)%n_phi]).T,
        np.array([Nbody+n_phi, Nbody_base+n_phi + (Nbody+1)%n_phi, Nbody_base + (Nbody+1)%n_phi]).T,
        np.array([Ntop, (Ntop+1)%n_phi, (len(verts)-2)*np.ones_like(Ntop)]).T,
        np.array([Nbottom, (len(verts)-1)*np.ones_like(Nbottom), (n_theta-1)*n_phi + (Nbottom+1)%n_phi]).T,
    ))

    return verts, faces

def _gen_mesh_cylinder(c1, c2, radius, n_height:int=3, n_phi:int=20):
    
    dc = (c2-c1)
    dc /= np.linalg.norm(dc)
    if np.allclose(dc, np.array([0,0,1])):
        x = np.cross(dc, np.array([1,0,0]))
    else:
        x = np.cross(dc, np.array([0,0,1]))
    x *= (radius/np.linalg.norm(x))
    
    y = np.cross(dc, x)
    y *= (radius/np.linalg.norm(y))

    H, Phi = np.meshgrid(np.arange(0,n_height+1)/n_height, np.arange(n_phi)*np.pi*2/n_phi, indexing='ij')
    Verts_raw = c1 + (c2-c1) * H[:,:,None] + np.cos(Phi)[:,:,None]*x + np.sin(Phi)[:,:,None]*y
    verts = Verts_raw.reshape((n_height+1)*n_phi, 3)
    
    # Unlike sphere, we don't have the top and bottom points. The order is opposite since we start from bottom.
    Nbody = np.arange(n_height*n_phi, dtype=int)
    Nbody_base = (Nbody//n_phi)*n_phi
    faces = np.concatenate((
        np.array([Nbody, Nbody_base + (Nbody+1)%n_phi, Nbody+n_phi]).T,
        np.array([Nbody+n_phi, Nbody_base + (Nbody+1)%n_phi, Nbody_base+n_phi + (Nbody+1)%n_phi]).T,
    ))

    return verts, faces

def _plot_molecule_ballstick(coords, bonds, bond_width, **kwargs):
    """ Plot ballstick graph. kwargs should be the processed scatter_kwargs as in `plot_molecule()`.
    """
    
    # In `plot_molecule()`, colorscale is explicitly used, so we are usually save to assume
    # the color names are convertable.
    colorlist = list(map(plotly.colors.unlabel_rgb, plotly.colors.convert_colors_to_same_type(list(kwargs['marker']['color']))[0]))

    verts, faces, colors = [], [], []
    
    for j in range(len(coords)):
        v, f = _gen_mesh_sphere(coords[j], kwargs['marker']['size'][j])
        verts.append(v)
        faces.append(f)
        colors.append(np.repeat([colorlist[j]], len(v), 0))

    if bond_width > 0:
        for p, q in bonds:
            v, f = _gen_mesh_cylinder(coords[p], coords[q], bond_width)
            verts.append(v)
            faces.append(f)
            colors.append(np.concatenate((
                np.repeat([colorlist[p]], len(v)//2, 0), 
                np.repeat([colorlist[q]], len(v)-len(v)//2, 0))))

    # squeez the mesh
    nverts = np.cumsum([0] + [len(v) for v in verts])
    verts = np.concatenate(verts, axis=0)
    faces = np.concatenate([f + n for f, n in zip(faces, nverts)], axis=0)
    colors = np.concatenate(colors)
    
    del kwargs['marker']

    # render
    return [
        go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2],
                  i=faces[:,0], j=faces[:,1], k=faces[:,2],
                  vertexcolor=colors,
                  **kwargs
                  )
    ]


def gen_bonds(charges:list, coords:np.ndarray, conn_cutoff:float=2.0, max_neighbor:int=6) -> np.ndarray:
    """ Generate a (N x 2) integer array for bonds.
    """
    
    dperiod = [np.searchsorted(constants.Periods, c) for c in charges]

    X, Y, Z = coords[:,0], coords[:,1], coords[:,2]
    D = np.sqrt((X[:,None]-X)**2+(Y[:,None]-Y)**2+(Z[:,None]-Z)**2)
    D[np.diag_indices_from(D)] = np.inf

    def get_max_neighbor(j):
        return min(max_neighbor, min(charges[j]-constants.Periods[dperiod[j]-1], constants.Periods[dperiod[j]]-charges[j])) if dperiod[j] != 0 else 1

    # we want to place bonds to 1) satisfy connection constraints and 2) choose the minimum distanced neightbor
    # we don't DFS since it still needs a "hard" criterion to choose the bond
    connect_to = [set() for j in range(len(coords))]
    neigh_idx = np.arange(len(coords))
    for j in range(len(coords)):
        sorted_idx = np.argsort(D[j,neigh_idx])
        neigh_sorted_idx = neigh_idx[sorted_idx]

        for k in range(min(len(neigh_sorted_idx), get_max_neighbor(j))):
            n = neigh_sorted_idx[k]
            if k > 0 and D[j,n] > conn_cutoff*D[j,neigh_sorted_idx[0]]:
                 break
            else:
                connect_to[j].add(n)

    pairs = set()
    for j in range(len(coords)):
        for c in connect_to[j]:
            if j in connect_to[c]:
                pairs.add((j, c) if j < c else (c, j))

    return np.array(list(pairs))
