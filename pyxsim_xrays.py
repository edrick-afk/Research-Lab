
# coding: utf-8

# # pyXSIM Simulation of a TNG Galaxy

# This notebook shows how we can use pyXSIM to create a mock X-ray observation of a galaxy from TNG50 (using also yt and SOXS).
# We tailor this example to the "Light Element Mapper" (LEM) x-ray telescope mission, but this can be changed to any other x-ray instrument supported by pyXSIM.

# First, we need to install pyXSIM into your Lab environment. Open a terminal (File -> New -> Terminal), and type:

# `pip install --user pyxsim`

# After a successful installation, we first import SOXS, which will download some data files:

# In[ ]:


from soxs.utils import soxs_cfg
soxs_cfg.set("soxs", "bkgnd_nH", "0.018") # avoid configparser error by specifying here
import soxs


# Then we import the remaining needed modules:

# In[ ]:


import yt
import pyxsim

import h5py
import numpy as np
import illustris_python as il

import os
from regions import RectangleSkyRegion
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import wcs
from astropy.io import fits


# ## Pick a galaxy, load the data, and prepare it for yt

# Specify the simulation, redshift, subhalo ID of a galaxy of interest:

# In[3]:


import illustris_python as il
import h5py
import numpy as np



basePath = "../sims.TNG/TNG100-1/output/"
snap = 99
haloID = 488530 #This is the corresponding subhalo ID for subhalo 477060 in z=95

halo = il.groupcat.loadSingle(basePath, snap, haloID=haloID)


# Then load the data, where we load from the entire FoF halo (and not just the central subhalo), to include as much surrounding material as easily possible:

# In[3]:


fields = ['Coordinates','GFM_CoolingRate','Density','InternalEnergy','ElectronAbundance','StarFormationRate']
gas = il.snapshot.loadHalo(basePath, snap, haloID, 'gas')
header = il.groupcat.loadHeader(basePath, snap)


# Then, since `yt` interfaces easiest with a "snapshot file", we quickly save this data to a temporary file:

# In[4]:


filename = "halo_%d.hdf5" % haloID
with h5py.File(filename,'w') as f:
    for key in gas.keys():
        f['PartType0/' + key] = gas[key]
        
    # some metadata that yt demands
    f.create_group('Header')
    f['Header'].attrs['NumFilesPerSnapshot'] = 1
    f['Header'].attrs['MassTable'] = np.zeros(6, dtype='float32')
    f['Header'].attrs['BoxSize'] = header['BoxSize']
    f['Header'].attrs['Time'] = header['Time']
    f['Header'].attrs['NumPart_ThisFile'] = np.array([gas['count'],0,0,0,0,0])


# ## Load the data into yt

# Then, we can load the data with `yt`:

# In[5]:


import yt

ds = yt.load(filename)


# Then, we make a phase space cut on the gas to make sure that we are focusing on the X-ray emitting gas. 
# The best way to do that is with a “particle filter” in yt.

# In[6]:


def hot_gas(pfilter, data):
    pfilter1 = data[pfilter.filtered_type, "temperature"] > 3.0e5
    pfilter2 = data["PartType0", "StarFormationRate"] == 0.0
    pfilter3 = data["PartType0", "GFM_CoolingRate"] < 0.0
    return (pfilter1 & pfilter2) & pfilter3

yt.add_particle_filter("hot_gas", function=hot_gas,
                       filtered_type='gas', requires=["temperature","density"])


# Add the particle filter to the dataset:

# In[7]:


ds.add_particle_filter("hot_gas")


# ## Quickly visualize the gas in the halo

# We can get the subhalo position from the info we grabbed earlier:

# In[8]:


c = ds.arr([halo["GroupPos"][0], halo["GroupPos"][1], halo["GroupPos"][2]], "code_length")


# Just to get a sense of what things look like, make a projection plot of the total gas density:

# In[9]:


prj = yt.ProjectionPlot(ds, "z", ("gas","density"), width=(0.4, "Mpc"), center=c)
prj.set_zlim(("gas","density"), 1.0e-7, 1.0)


# And show the same plot of the "hot" gas density:

# In[84]:


prj = yt.ProjectionPlot(ds, "z", ("hot_gas","density"), width=(0.4, "Mpc"), center=c)
prj.set_zlim(("hot_gas","density"), 1.0e-7, 1.0e-3)


# ## Configure the x-ray emission, and observation, models

# Now, in order to make the mock observation, we have to set up a `ThermalSourceModel` in pyXSIM, telling it which fields to use from the dataset, and the min, max, and binning of the spectrum:

# In[86]:


emin = 0.05
emax = 4.0
nbins = 4000
source_model = pyxsim.ThermalSourceModel(
    "apec", emin, emax, nbins, ("hot_gas","metallicity"),
    temperature_field=("hot_gas","temperature"),
    emission_measure_field=("hot_gas", "emission_measure"),
)


# Now we specify fiducial values of the exposure time, collecting area, and redshift of the object. NOTE that the "area" here is not the effective area of the telescope--this is just a parameter that we need to decide how many sample photons to make. This number should be bigger than the peak of the telescope+instrument effective area curve.

# In[87]:


exp_time = (500., "ks") # exposure time
area = (5000.0, "cm**2") # collecting area
redshift = 0.025


# Now create a box centered on the galaxy, 1 Mpc in width, which will be used to draw the Arepo cells to make the photons.

# In[88]:


width = ds.quan(1.0, "Mpc")
le = c - 0.5*width
re = c + 0.5*width
box = ds.box(le, re)


# ## Generate the mock x-ray emission

# Now we actually make the photons:

# In[89]:


n_photons, n_cells = pyxsim.make_photons(f"halo_{haloID}_photons", box, redshift, area, exp_time, source_model)


# Now we project the photons, also including foreground galactic absorption. We project along the "z"-axis of the simulation box.

# In[73]:


n_events = pyxsim.project_photons(f"halo_{haloID}_photons", f"halo_{haloID}_events", "z", (45.,30.),
                                  absorb_model="wabs", nH=0.01)


# Open the file containing the projected events, and convert it to SIMPUT format. 

# In[74]:


events = pyxsim.EventList(f"halo_{haloID}_events.h5")
events.write_to_simput(f"halo_{haloID}", overwrite=True)


# ## Create a synthetic observation of these x-rays with LEM

# We can now use this SIMPUT catalog to make a mock LEM observation. For now, we turn off the astrophysical background from the galaxy, and distant point sources.

# In[75]:


instrument = "lem"
soxs.instrument_simulator(f"halo_{haloID}_simput.fits", f"halo_{haloID}_evt.fits", (500.0, "ks"), instrument, (45.,30.), overwrite=True, foreground=False, ptsrc_bkgnd=False)


# This produces an event file, which we now convert to an image and show. The green square in the image shows the FOV of LEM. 

# In[76]:


soxs.write_image(f"halo_{haloID}_evt.fits", f"halo_{haloID}_img.fits", emin=0.1, emax=2.0, overwrite=True)
center_sky = SkyCoord(45, 30, unit='deg', frame='fk5')
region_sky = RectangleSkyRegion(center=center_sky, width=32 * u.arcmin, height=32*u.arcmin)
with fits.open(f"halo_{haloID}_img.fits") as f:
    w = wcs.WCS(header=f[0].header)
    fig, ax = soxs.plot_image(f"halo_{haloID}_img.fits", stretch='log', cmap='afmhot', vmax=1500.0, width=0.6)
ax.add_artist(region_sky.to_pixel(w).as_artist())


# We can also make and plot a spectrum:

# In[77]:


soxs.write_spectrum(f"halo_{haloID}_evt.fits", f"halo_{haloID}_evt.pi", overwrite=True)
fig, ax = soxs.plot_spectrum(f"halo_{haloID}_evt.pi", xmin=0.3, xmax=1.0, xscale="linear")


# That's it! Thanks to John ZuHone for preparing the original walkthrough for the LEM workshop in Feb 2022.
# 
# Please address any questions/comments to John ZuHone and Dylan Nelson.
