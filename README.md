
# Cube

Simple visualizer of [cube files](paulbourke.net/dataformats/cube/). Jupyter notebook compatible.

<div style="display:flex; flex-direction: row; justify-content: center; align-items: center">
<img width="30%" height="20%" src="doc/0.png">
<img width="30%" height="20%" src="doc/1.png">
<img width="30%" height="20%" src="doc/2.png">
</div>


## Installation

    pip install -e git+https://github.com/yanzewu/cube

Prerequesties:
- plotly
- scikit-image (optional)

## Usage

The simplist way is 

    import cube
    cube.plot_cube('my_cube_file').show()

More example can be found in [example.ipynb](doc/example.ipynb).
