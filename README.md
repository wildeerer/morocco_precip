# ML Precipitation Forecasting in Morocco

The goal of this project is to implement a neural network that will upscale the data we have and use it to improve precipitation forecasting. \
The neural network being implemented was derived from this source: https://gmd.copernicus.org/articles/16/535/2023/#section7


## Requirements installation
To install requirements needed to run this project run the command 
```pip install path/to/requirements.txt```. If you're running this program from the parent directory you can just run ```pip install requirements.txt```. If you have cuda available (Windows and Linux only) consider installing the cuda version of pytorch.\
\
The folder structure of the project is split into a network, preprocessing, notebooks, and data directory (TODO).

```new_test.ipynb``` : contains the working neural network.\
```test.ipynb```: contains the random forest implementation.


## Data Source
https://portal.nccs.nasa.gov/datashare/fldas-forecast/Files/

## Resources:
https://github.com/javedali99/python-resources-for-earth-sciences 


## TODO
- [ ] CLEAN UP CODE!!!!
- [ ] Define project goals
- [ ] Go over data needs
- [ ] More comprehensive data exploration, pre and post NN transformation
- [ ] Find better Earth science visualization libraries/programs
- [ ] Improve folder structure
- [ ] Add notes and resources into notebooks