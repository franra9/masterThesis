{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Example workflow to model the mass-balance of the Aneto-glacier:\n",
    "\n",
    "-> you need to install the MBsandbox to be able to run the code (you also need the most recent OGGM version):\n",
    "  - see: https://github.com/OGGM/massbalance-sandbox#how-to-install\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# I have made a change in \n",
    "# ~/.local/lib/python3.8/site-packages/pandas/_typing.py\n",
    "#(np.random.BitGenerator --> np.random.bit_generator.BitGenerator)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.17.4\n"
     ]
    },
    {
     "ename": "ImportError",
     "evalue": "this version of pandas is incompatible with numpy < 1.18.5\nyour numpy version is 1.17.4.\nPlease upgrade numpy to >= 1.18.5 to use this pandas version",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-4-837e8cd69d3d>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msystem\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'mamba install numpy'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__version__\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      6\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mxarray\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mxr\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mseaborn\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0msns\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.8/site-packages/pandas/__init__.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     20\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     21\u001b[0m \u001b[0;31m# numpy compat\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 22\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mpandas\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcompat\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mis_numpy_dev\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0m_is_numpy_dev\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     23\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     24\u001b[0m \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.8/site-packages/pandas/compat/__init__.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     13\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mpandas\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_typing\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mF\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 15\u001b[0;31m from pandas.compat.numpy import (\n\u001b[0m\u001b[1;32m     16\u001b[0m     \u001b[0mis_numpy_dev\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     17\u001b[0m     \u001b[0mnp_version_under1p19\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.8/site-packages/pandas/compat/numpy/__init__.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     21\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0m_nlv\u001b[0m \u001b[0;34m<\u001b[0m \u001b[0mVersion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m_min_numpy_ver\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 23\u001b[0;31m     raise ImportError(\n\u001b[0m\u001b[1;32m     24\u001b[0m         \u001b[0;34mf\"this version of pandas is incompatible with numpy < {_min_numpy_ver}\\n\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     25\u001b[0m         \u001b[0;34mf\"your numpy version is {_np_version}.\\n\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mImportError\u001b[0m: this version of pandas is incompatible with numpy < 1.18.5\nyour numpy version is 1.17.4.\nPlease upgrade numpy to >= 1.18.5 to use this pandas version"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import os\n",
    "#os.system('mamba install numpy')\n",
    "print(np.__version__)\n",
    "import pandas as pd\n",
    "import xarray as xr\n",
    "import seaborn as sns\n",
    "import pickle\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "\n",
    "from numpy.testing import assert_allclose\n",
    "import scipy\n",
    "import scipy.stats as stats\n",
    "from IPython.core.pylabtools import figsize\n",
    "import os\n",
    "\n",
    "import oggm\n",
    "from oggm import cfg, utils, workflow, tasks, graphics, entity_task\n",
    "from oggm.utils import date_to_floatyear\n",
    "from oggm.shop import gcm_climate\n",
    "from oggm.core import massbalance, flowline, climate\n",
    "\n",
    "import logging\n",
    "log = logging.getLogger(__name__)\n",
    "\n",
    "cfg.initialize() #logging_level='WARNING'\n",
    "cfg.PARAMS['use_multiprocessing'] = False\n",
    "cfg.PARAMS['continue_on_error'] = False\n",
    "cfg.PATHS['working_dir'] = utils.gettempdir(dirname='OGGM-sfc-type', reset=False)\n",
    "\n",
    "# use Huss flowlines\n",
    "base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'\n",
    "            'L1-L2_files/elev_bands')\n",
    "\n",
    "# import the MBsandbox modules\n",
    "from MBsandbox.mbmod_daily_oneflowline import (process_w5e5_data, BASENAMES, MultipleFlowlineMassBalance_TIModel, TIModel_Sfc_Type)\n",
    "from MBsandbox.help_func import minimize_bias_geodetic, minimize_winter_mb_brentq_geod_via_pf\n",
    "\n",
    "cfg.PARAMS['hydro_month_nh']=1  # to calibrate to the geodetic estimates we need calendar years !!!\n",
    "\n",
    "df = ['RGI60-11.03208']  # here we select Aneto glacier!\n",
    "\n",
    "gdirs = workflow.init_glacier_directories(df, from_prepro_level=2,\n",
    "                                              prepro_border=10,\n",
    "                                              prepro_base_url=base_url,\n",
    "                                              prepro_rgi_version='62')\n",
    "gdir = gdirs[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# this is the applied resolution of the precipitation and temperature climate dataset\n",
    "temporal_resol = 'daily'\n",
    "baseline_climate = 'W5E5' \n",
    "process_w5e5_data(gdir, temporal_resol=temporal_resol,\n",
    "                  climate_type=baseline_climate)\n",
    "\n",
    "# let's get the heights and widths of the inversion (we use here the elevation-band flowline!)\n",
    "h, w = gdir.get_inversion_flowline_hw()\n",
    "fls = gdir.read_pickle('inversion_flowlines')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# just show some basic informations of the glacier:\n",
    "gdir"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ok, apparently the glacier is not a reference glacier according to WGMS (no annual MB exist)\n",
    "gdir.get_ref_mb_data(input_filesuffix='_daily_W5E5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# MB model options: \n",
    "# we just use the most complicated ones: \n",
    "mb_type = 'mb_real_daily'  # daily climate resolution \n",
    "grad_type = 'var_an_cycle'  # a variable temperature lapse rate (cte between the years but changing spatially and between the months)\n",
    "melt_f_update = 'monthly'  # how often the melt factor is updated to distinguish between different snow / firn ages\n",
    "melt_f_change = 'neg_exp'  # the way how the melt factor changes over time \n",
    "tau_e_fold_yr = 1  # how fast the melt factor of snow converges to the ice melt factor \n",
    "kwargs_for_TIModel_Sfc_Type = {'melt_f_update':melt_f_update, 'melt_f_change': melt_f_change, 'tau_e_fold_yr':tau_e_fold_yr}\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "These are the \"free\" parameters of the MB model, at the moment, I have just set the prcp. factor to 1.5, do not use any temp. bias (default option) and I use an arbitrarily chosen melt factor of 200 (no calibration done so far!)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "melt_f = 200\n",
    "pf = 1.5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# with normal spinup for 6 years\n",
    "mb_mod_monthly_0_5_m = TIModel_Sfc_Type(gdir, melt_f, mb_type=mb_type, grad_type = grad_type,\n",
    "                                        prcp_fac=pf,\n",
    "                                        melt_f_update=melt_f_update,\n",
    "                                        melt_f_change = melt_f_change, \n",
    "                                        tau_e_fold_yr = tau_e_fold_yr,\n",
    "                                        baseline_climate=baseline_climate)\n",
    "# default temp. bias is 0 !\n",
    "mb_mod_monthly_0_5_m.temp_bias"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mb_mod_monthly_0_5_m.prcp_fac"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mb_mod_monthly_0_5_m.melt_f"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, we can calibrate the melt factor (same as mu_star in OGGM default) to match e.g. the geodetic MB estimate (mean specific MB of 2000-2020) of Hugonnet et al. (2021). "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# first get the observed geodetic data for calibration\n",
    "pd_geodetic = utils.get_geodetic_mb_dataframe()\n",
    "pd_geodetic = pd_geodetic.loc[pd_geodetic.period == '2000-01-01_2020-01-01']\n",
    "mb_geodetic = pd_geodetic.loc[df].dmdtda.values * 1000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "years = np.arange(2000, 2020, 1)\n",
    "melt_f_opt_0_5_m = scipy.optimize.brentq(minimize_bias_geodetic, \n",
    "                                         10, 1000, # minimum and maximum value of melt_f\n",
    "                                         xtol=0.01, args=(mb_mod_monthly_0_5_m,\n",
    "                                                      mb_geodetic,\n",
    "                                                      h, w, pf, False,\n",
    "                                                      years,\n",
    "                                                      False, True, # yes, do spinup before\n",
    "                                                      ), disp=True)\n",
    "# change the melt_f to the newly calibrated one\n",
    "mb_mod_monthly_0_5_m.melt_f = melt_f_opt_0_5_m\n",
    "spec_0_5_m = mb_mod_monthly_0_5_m.get_specific_mb(year=years, fls=fls)\n",
    "# check if the calibration has worked as we expect:\n",
    "np.testing.assert_allclose(spec_0_5_m.mean(), mb_geodetic, rtol = 1e-4)\n",
    "print(melt_f_opt_0_5_m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# this is the \"raw\" climate that is used for the glacier during the calibration time period (without local downscaling by applying a multiplicative precipitation correction factor or a temperature bias)\n",
    "fpath_climate = gdir.get_filepath('climate_historical', filesuffix='_daily_W5E5')\n",
    "ds_clim = xr.open_dataset(fpath_climate)\n",
    "ds_clim"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So far, we just applied an arbitrary precipitation factor. You could calibrate the precipitation factor to match other observations, for example the winter mass-balance, by minimizing both, geodetic bias and winter MB, using melt_f and prcp-fac as parameters to calibrate (via `help_func/minimize_winter_mb_brentq_geod_via_pf`). However, as your glacier is no reference glacier, you can not use this approach!\n",
    "\n",
    "Instead you could use a relationship that was found for reference glaciers that have winter MB available:\n",
    "- glaciers with stronger winter precipitation do have rather larger precipitation factors\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# we are in the NH, so to get winter prcp we need October to April\n",
    "ds_prcp_winter = ds_clim.prcp.where(ds_clim.prcp['time.month'].isin([10, 11, 12, 1, 2, 3, 4]),\n",
    "                                                           drop=True).mean().values\n",
    "# mean winter prcp in kg m-2, mean over 1979-2020\n",
    "ds_prcp_winter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "a_log_multiplied = -0.988689\n",
    "b_intercept = 4.004772\n",
    "def log_func(x, a, b):\n",
    "    r = a*np.log(x) +b\n",
    "    # don't allow extremely low/high prcp. factors!!!\n",
    "    if np.shape(r) == ():\n",
    "        if r > 10:\n",
    "            r = 10\n",
    "        if r<0.1:\n",
    "            r=0.1\n",
    "    else:\n",
    "        r[r>10] = 10\n",
    "        r[r<0.1] = 0.1\n",
    "    return r\n",
    "\n",
    "winter_prcp_values = np.arange(0.1, 20,0.05)\n",
    "plt.plot(winter_prcp_values, log_func(winter_prcp_values, a_log_multiplied, b_intercept), label='fitted precipitation factor relation to winter\\nprecipitation (found from 114 reference\\nglaciers where winter MB was matched)')\n",
    "fit_prcp_fac_aneto = log_func(ds_prcp_winter, a_log_multiplied, b_intercept)\n",
    "plt.axvline(ds_prcp_winter, ls=':', color='grey', label=f'Aneto glacier, fitted prcp_fac={fit_prcp_fac_aneto:.2f}')\n",
    "plt.axhline(fit_prcp_fac_aneto, ls=':', color='grey')\n",
    "plt.ylabel('fitted precipitation factor')\n",
    "plt.xlabel('mean daily winter precipitation (kg m-2)') # mean over 1979-2020\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# recalibrate melt_f with the new pf!\n",
    "pf = fit_prcp_fac_aneto\n",
    "melt_f_opt_0_5_m = scipy.optimize.brentq(minimize_bias_geodetic, 10, 1000,\n",
    "                                     xtol=0.01, args=(mb_mod_monthly_0_5_m,\n",
    "                                                      mb_geodetic,\n",
    "                                                      h, w, pf, False,\n",
    "                                                      years,\n",
    "                                                      False, True, # do spinup before\n",
    "                                                      ), disp=True)\n",
    "# change the melt_f to the newly calibrated one\n",
    "mb_mod_monthly_0_5_m.melt_f = melt_f_opt_0_5_m\n",
    "spec_0_5_m = mb_mod_monthly_0_5_m.get_specific_mb(year=years, fls=fls)\n",
    "# check if the calibration has worked as we expect:\n",
    "np.testing.assert_allclose(spec_0_5_m.mean(), mb_geodetic, rtol = 1e-4)\n",
    "print(melt_f_opt_0_5_m, pf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(years, spec_0_5_m)\n",
    "plt.ylabel('specific mass-balance')\n",
    "plt.xlabel('years')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Some more details about the applied surface type distinction method to get a melt factor that distinguishes between ice and snow age:\n",
    "\n",
    "inside of `TIModel_Sfc_Type`, there is **a surface type distinction model included with a bucket system together with a melt_f that varies with age** :\n",
    "- there are two options included at the moment:\n",
    "    - `melt_f_update=annual`\n",
    "        - If annual, then it uses 1 snow\n",
    "            and 5 firn buckets with yearly melt factor updates.\n",
    "    - `melt_f_update=monthly`:\n",
    "        -  If monthly, each month the snow is ageing over 6 years (i.e., 72 months -> 72 buckets).\n",
    "    - the ice bucket is thought as an \"infinite\" bucket (because we do not know the ice thickness at this model stage)\n",
    "    - Melt factors are interpolated either:\n",
    "        - linearly inbetween the buckets.\n",
    "        - or using a negativ exponential change assumption with an e-folding change assumption of e.g. 0.5 or 1 year\n",
    "- default is to use a **spinup** of 6 years. So to compute the specific mass balance between 2000 and 2020, with `spinup=True`, the annual mb is computed since 1994 where at first everything is ice, and then it accumulates over the next years, so that in 2000 there is something in each bucket ...\n",
    "    - if we start in 1979 (start of W5E5), we neglect the spinup because we don't have climate data before 1979\n",
    "\n",
    "- the ratio of snow melt factor to ice melt factor is set to 0.5 (as in GloGEM) but it can be changed via `melt_f_ratio_snow_to_ice`\n",
    "    - if we set `melt_f_ratio_snow_to_ice=1` the melt factor is equal for all buckets, hence the results are equal to no surface type distinction (as in `TIModel`)\n",
    "- `get_annual_mb` and `get_monthly_mb` work as in PastMassBalance, however they only accept the height array that corresponds to the inversion height (so no mass-balance elevation feedback can be included at the moment!)\n",
    "    - that means the given mass-balance ist the mass-balance over the inversion heights (before doing the inversion and so on)\n",
    "- the buckets are automatically updated when using `get_annual_mb` or `get_monthly_mb` via the `TIModel_Sfc_Type.pd_bucket` dataframe \n",
    "- to make sure that we do not compute mass-balance twice and to always have a spin-up of 6 years, the mass balance is automatically saved under \n",
    "    - `get_annual_mb.pd_mb_annual`: for each year\n",
    "        - when using `get_monthly_mb` for several years, after computing the December month, the `pd_mb_annual` dataframe is updated\n",
    "    - `get_annual_mb.pd_mb_monthly`: for each month \n",
    "        - note that this stays empty if we only use get_annual_mb with annual melt_f_update\n",
    "        \n",
    "        \n",
    "**-> you will use melt_f_update=monthly, spinup=True, melt_f_ratio_snow_to_ice=0.5, a negativ exponential change assumption with an e-folding change assumption of 1 year (see more details in the following plots below)** "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Monthly MB is saved in TIModel_Sfc_Type instance:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# the first 6 years are the spinup\n",
    "mb_mod_monthly_0_5_m.pd_mb_monthly\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Also annual MB is saved in TIModel_Sfc_Type instance:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# the first 6 years are the spinup\n",
    "mb_mod_monthly_0_5_m.pd_mb_annual\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Buckets when using monthly melt_f_update:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# this is the output after updating the surface type (and before the next month)\n",
    "# hence: the \"youngest\" bucket 0 is 0 (because already updated to next older month!)\n",
    "mb_mod_monthly_0_5_m.pd_bucket "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Example plot with monthly melt_f update:**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mb = mb_mod_monthly_0_5_m\n",
    "name = '0_5_m_neg_exp_tau1yr'\n",
    "mb.get_specific_mb(year=years, fls=fls)\n",
    "mb_annual_dict_name = {}\n",
    "for y in years:\n",
    "    mb_y = mb.pd_mb_annual[y]  # output is in m of ice per second ! (would be the same as doing `mb.get_annual_mb(h, y)`)\n",
    "    mb_y = mb_y * mb.SEC_IN_YEAR * mb.rho\n",
    "    mb_gradient,_,_,_,_ = scipy.stats.linregress(h[mb_y<0], y=mb_y[mb_y<0]) \n",
    "    mb_annual_dict_name[y] = mb_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.rcParams.update({'font.size': 20})\n",
    "plt.figure(figsize=(24, 8))\n",
    "lw=2\n",
    "plt.subplot(121)\n",
    "plt.title('melt_f change with ageing surface\\nfrom snow to firn to ice')\n",
    "ax = plt.gca()\n",
    "plt.text(-0.15,1.02,'(a)', transform=ax.transAxes)\n",
    "\n",
    "meltis = []\n",
    "for _,m in mb_mod_monthly_0_5_m.melt_f_buckets.items():\n",
    "    meltis.append(m)\n",
    "for m in np.arange(0,12):\n",
    "    meltis.append(mb_mod_monthly_0_5_m.melt_f)\n",
    "    \n",
    "plt.plot(np.arange(0,7.01,1/12), meltis, color='red', label='sfc type distinction, monthly update\\nneg_exp tau=1yr,\\nmelt_f of ice={:1.0f}, prcp_fac={:.2f}'.format(melt_f_opt_0_5_m, fit_prcp_fac_aneto))\n",
    "# plt.plot(np.arange(0,7.01,1/12), meltis, color='red', label='sfc type distinction, monthly update\\nexp. change (tau=1yr), melt_f={:0.1f}'.format(melt_f_opt_0_5_m_neg_exp_tau1yr))\n",
    "plt.xticks(np.arange(0,7.1,1))\n",
    "\n",
    "\n",
    "plt.xlabel('snow age (in years)')\n",
    "#plt.yticks(np.arange(0,1250,300,350,400,450,500,550])\n",
    "plt.ylabel(r'melt_f (mm w.e. K$^{-1}$ mth$^{-1}$)')\n",
    "\n",
    "handles, labels = ax.get_legend_handles_labels()\n",
    "# sort both labels and handles by labels\n",
    "labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))\n",
    "ax.legend(handles, labels)\n",
    "plt.axvline(6, color='grey', ls='--', alpha=0.5)\n",
    "\n",
    "plt.subplot(122)\n",
    "ax = plt.gca()\n",
    "plt.text(-0.15,1.02,'(b)', transform=ax.transAxes)\n",
    "plt.plot(pd.DataFrame(mb_annual_dict_name).mean(axis=1).values,\n",
    "         h, color = 'blue', \n",
    "         lw=lw)\n",
    "\n",
    "plt.title('mean MB profile (here: 2000-2019)')\n",
    "plt.xlabel('kg m-2')\n",
    "plt.ylabel('altitude (m)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# can also just look at seasonal MB profiles\n",
    "mb.reset_pd_mb_bucket()\n",
    "mb_grad_dict_name = []\n",
    "mb_monthly_dict_name = {}\n",
    "bucket_name = {}\n",
    "for y in years:\n",
    "    for m in np.arange(1,13,1):\n",
    "        floatyr = date_to_floatyear(y,m)\n",
    "        #if name != '0_5_m':\n",
    "        _, bucket_name[floatyr] = mb.get_monthly_mb(h, year=floatyr, bucket_output =True)\n",
    "        mb_m = mb.pd_mb_monthly[floatyr]  # output is in m of ice per second !\n",
    "        try:\n",
    "            mb_gradient,_,_,_,_ = scipy.stats.linregress(h[mb_m<0], y=mb_m[mb_m<0]) \n",
    "        except:\n",
    "            mb_gradient = np.NaN\n",
    "        mb_monthly_dict_name[floatyr] = mb_m"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## also get winter / summer MB profiles\n",
    "y0 = 1980 \n",
    "m_winter_mb = [10,11,12,1,2,3,4]\n",
    "m_summer_mb = [5,6,7,8,9]\n",
    "yr_changes = True\n",
    "m_start = 10\n",
    "add_climate = True\n",
    "ratio = 1\n",
    "melt_m_10 = {}\n",
    "melt_m_4 = {}\n",
    "melt_m_5 = {}\n",
    "mb_grad_winter = {}\n",
    "mb_grad_summer = {}\n",
    "mb_grad_winter_melt = {}\n",
    "mb_grad_summer_melt = {}\n",
    "color_dict = {'1_a': 'black', '0_5_m':'blue', '0_5_m_neg_exp':'red'}\n",
    "fig_w, axs_w = plt.subplots(1,7, figsize=(24,8), sharey=True, sharex=True)\n",
    "plt.suptitle('monthly MB profiles')\n",
    "\n",
    "fig_s, axs_s = plt.subplots(1,7, figsize=(24,8), sharey=True, sharex=True)\n",
    "\n",
    "mb.reset_pd_mb_bucket()\n",
    "_ = mb.get_specific_mb(h, widths=w, year=np.arange(1979,2020,1))\n",
    "\n",
    "pd_winter_mb = pd.DataFrame(index=h, columns=np.arange(y0,2020,1))\n",
    "pd_summer_mb = pd.DataFrame(index=h, columns=np.arange(y0,2020,1))\n",
    "pd_winter_melt = pd.DataFrame(index=h, columns=np.arange(y0,2020,1))\n",
    "pd_summer_melt = pd.DataFrame(index=h, columns=np.arange(y0,2020,1))\n",
    "# Let's plot monthly MB profiles of that year!!\n",
    "for year in np.arange(y0, 2020,1):\n",
    "    mbs_winter = 0\n",
    "    mbs_summer = 0\n",
    "    prcp_sol_winter = 0\n",
    "    prcp_sol_summer = 0\n",
    "\n",
    "    for j,m in enumerate(m_winter_mb):\n",
    "        if (m in np.arange(m_start, 13, 1)) and (yr_changes):\n",
    "            floatyr = utils.date_to_floatyear(year-1, m)\n",
    "        else:\n",
    "            floatyr = utils.date_to_floatyear(year, m)\n",
    "        out = mb.get_monthly_mb(h, year=floatyr, add_climate=True)\n",
    "        out, t, tfm, prcp, prcp_sol = out\n",
    "        mbs_winter += out * ratio\n",
    "        prcp_sol_winter += prcp_sol *ratio\n",
    "        if (year== 2008 and m>=10) or (year == 2009 and m<10):\n",
    "            try:\n",
    "                if m == 10 and year == 2008:\n",
    "                    melt_m_10[name] = out*mb.SEC_IN_MONTH * mb.rho-prcp_sol\n",
    "                elif m == 4 and year == 2009:\n",
    "                    melt_m_4[name] = out*mb.SEC_IN_MONTH * mb.rho-prcp_sol\n",
    "                axs_w[j].plot(out*mb.SEC_IN_MONTH *mb.rho, h)\n",
    "                axs_w[j].set_title(f'month: {m}')\n",
    "            except:\n",
    "                pass\n",
    "        #mbs_winter += mb_winter_m\n",
    "    for jj, m in enumerate(m_summer_mb):\n",
    "        floatyr = utils.date_to_floatyear(year, m)\n",
    "        out = mb.get_monthly_mb(h, year=floatyr, add_climate=True)\n",
    "        out, t, tfm, prcp, prcp_sol = out\n",
    "        mbs_summer += out * ratio\n",
    "        prcp_sol_summer += prcp_sol *ratio\n",
    "        if (year== 2008 and m>=10) or (year == 2009 and m<10):\n",
    "            try:\n",
    "                if m == 5 and year == 2009:\n",
    "                    melt_m_5[name] = out*mb.SEC_IN_MONTH * mb.rho-prcp_sol\n",
    "            except:\n",
    "                pass\n",
    "        if year == 2009:\n",
    "            try:\n",
    "                axs_s[jj].plot(out*mb.SEC_IN_MONTH * mb.rho, h)\n",
    "                axs_s[jj].set_title(f'month: {m}')\n",
    "            except:\n",
    "                pass\n",
    "\n",
    "\n",
    "    mbs_winter = mbs_winter * mb.SEC_IN_MONTH * mb.rho\n",
    "    mbs_summer = mbs_summer * mb.SEC_IN_MONTH * mb.rho\n",
    "\n",
    "    pd_winter_mb.loc[h, year] = mbs_winter\n",
    "    pd_summer_mb.loc[h, year] = mbs_summer\n",
    "\n",
    "    pd_winter_melt.loc[h, year] = mbs_winter-prcp_sol_winter\n",
    "    pd_summer_melt.loc[h, year] = mbs_summer-prcp_sol_summer\n",
    "\n",
    "\n",
    "\n",
    "mb_grad_winter[name] = pd_winter_mb.mean(axis=1).values, pd_winter_mb.mean(axis=1).index\n",
    "mb_grad_summer[name] = pd_summer_mb.mean(axis=1).values, pd_summer_mb.mean(axis=1).index\n",
    "\n",
    "mb_grad_winter_melt[name] = pd_winter_melt.mean(axis=1).values, pd_winter_melt.mean(axis=1).index\n",
    "mb_grad_summer_melt[name] = pd_summer_melt.mean(axis=1).values, pd_summer_melt.mean(axis=1).index\n",
    "\n",
    "axs_s[0].set_ylabel('altitude (m)')\n",
    "axs_w[0].set_ylabel('altitude (m)')\n",
    "axs_s[0].set_xlabel('kg m-2')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You can get the monthly climate like that:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "m=1\n",
    "temp, tfm, prcp, prcp_solid = mb.get_monthly_climate(h, year=utils.date_to_floatyear(2009, m))\n",
    "temp, tfm, prcp, prcp_solid"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**One year of snow ageing on the glacier (here for 2008-10 to 2009-09)**:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "name = '0_5_a'\n",
    "fig, axs = plt.subplots(4, 3, #gridspec_kw={'width_ratios': [1,1,1]},\n",
    "                        figsize=(24, 24),\n",
    "                       constrained_layout=True, sharey=True)\n",
    "\n",
    "for j,m in enumerate([10,11,12,1,2,3,4,5,6,7,8,9]):\n",
    "    if m in [10,11,12]:\n",
    "        k = 0\n",
    "    elif m in [1,2,3]:\n",
    "        k=1\n",
    "        j = j-3\n",
    "    elif m in [4,5,6]:\n",
    "        k=2\n",
    "        j = j-6\n",
    "    else:\n",
    "        k=3\n",
    "        j=j-9\n",
    "    ax = axs[k,j]\n",
    "    ax.set_facecolor('gainsboro')\n",
    "    if m >= 10:\n",
    "        year = 2008#9 # 2001\n",
    "    else:\n",
    "        year = 2009 #10 # 2000\n",
    "\n",
    "    floatyr = utils.date_to_floatyear(year, m)\n",
    "    \n",
    "    sns_pd_bucket_sel = bucket_name[floatyr].copy()\n",
    "    sns_pd_bucket_sel['altitude (m)'] = h.round(1)\n",
    "    sns_pd_bucket_sel = sns_pd_bucket_sel[sns_pd_bucket_sel.columns[::-1]]\n",
    "    \n",
    "    sns_pd_bucket_sel.index = bucket_name[floatyr].index #sns_pd_bucket_sel['altitude (m)']\n",
    "    # only plot every second altitude band !!! \n",
    "    sns_pd_bucket_sel = sns_pd_bucket_sel.iloc[::2]\n",
    "    pd_pivot = sns_pd_bucket_sel[sns_pd_bucket_sel.columns[2:]]\n",
    "    pd_pivot.index = pd_pivot.index/1000\n",
    "    pd_pivot = pd_pivot.sort_index(ascending=False)\n",
    "\n",
    "    pd_pivot.plot.barh(stacked=True, ax= ax,\n",
    "                       colormap='Blues_r', \n",
    "                       )\n",
    "\n",
    "    han, lab = ax.get_legend_handles_labels()\n",
    "    ax.get_legend().remove()\n",
    "    if j == 0:\n",
    "        ax.legend(handles = [han[k-1] for k in [12, 24, 36, 48, 60, 72]],\n",
    "                  labels = [lab[k-1] for k in [12, 24, 36, 48, 60, 72]],\n",
    "                  title='snow age (<72 months)',  loc = 4, framealpha = 0.4, ncol=6,\n",
    "                  bbox_to_anchor=(1,0.05),\n",
    "                  labelspacing=0.1, handlelength=1, handletextpad=0.3, columnspacing=0.9);\n",
    "        ax.set_ylabel('distance along flowline (km)')\n",
    "    \n",
    "    ax.set_xlabel(r'firn or snow above ice (kg m$^{-2}$)')\n",
    "    ax.set_title(f'm={m}', fontsize=18)\n",
    "    if m>= 10:\n",
    "        ax.text(0.77,0.012, 'year=2008', transform=ax.transAxes)\n",
    "    else:\n",
    "        ax.text(0.77,0.012, 'year=2009', transform=ax.transAxes)\n",
    "    if j == 2:\n",
    "        ax2 = ax.twinx()\n",
    "        pd_pivot.plot.barh(stacked=True, ax= ax2,\n",
    "                   colormap='Blues_r', alpha = 0,\n",
    "                   )\n",
    "        ax2.set_yticklabels(h[::2].round(0).astype(int)[::-1])\n",
    "        ax2.set_ylabel('altitude (m)')\n",
    "        ax2.get_legend().remove()\n",
    "    ax.set_xlim([0,2500])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Inversion with Glen-A calibration\n",
    "\n",
    "(to match ice thickness estimate of Farinotti(2020))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Computes the Flowline along the unglaciated downstream topography\n",
    "tasks.compute_downstream_line(gdir)\n",
    "# The bedshape obtained by fitting a parabola to the line’s normals. Also computes the downstream’s altitude.\n",
    "tasks.compute_downstream_bedshape(gdir)\n",
    "\n",
    "# get the apparent_mb (necessary for inversion)\n",
    "ye = 2020\n",
    "climate.apparent_mb_from_any_mb(gdir, mb_model=mb,\n",
    "                                mb_years=np.arange(2000, ye, 1))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# here glen-a is calibrated to match gdirs glaciers in total (as there is only one glacier, it just matches perfectly the one glacier!)\n",
    "border = 80\n",
    "filter = border >= 20\n",
    "pd_inv_melt_f = oggm.workflow.calibrate_inversion_from_consensus(gdirs,\n",
    "                                                      apply_fs_on_mismatch=False,ignore_missing=False,\n",
    "                                                      error_on_mismatch=True,\n",
    "                                                      filter_inversion_output=filter)\n",
    "# so for init_present_time_glacier, automatically the new glen a volume inversion is used!\n",
    "workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# double check: volume of Farinotti estimate is equal to calibrated oggm estimate\n",
    "# so estimates by Farinotti exist for each glacier individually?\n",
    "np.testing.assert_allclose(pd_inv_melt_f.sum()['vol_itmix_m3'], pd_inv_melt_f.sum()['vol_oggm_m3'], rtol = 1e-2)\n",
    "pd_inv_melt_f"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Get fixed geometry MB:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kwargs = kwargs_for_TIModel_Sfc_Type.copy()\n",
    "kwargs['mb_type'] = mb_type\n",
    "kwargs['grad_type'] = grad_type\n",
    "#kwargs['melt_f'] = melt_f_opt_0_5_m\n",
    "#kwargs['prcp_fac'] = fit_prcp_fac_aneto\n",
    "fs_new = '_{}_sfc_type_{}_{}_{}'.format(baseline_climate, melt_f_change, mb_type, grad_type)\n",
    "d = {'melt_f': melt_f_opt_0_5_m,\n",
    "     'pf': fit_prcp_fac_aneto,\n",
    "     'temp_bias': 0}\n",
    "gdir.write_json(d, filename='melt_f_geod', filesuffix=fs_new)\n",
    "\n",
    "from MBsandbox.mbmod_daily_oneflowline import compile_fixed_geometry_mass_balance_TIModel\n",
    "#filesuffix = f'_gcm_{ensemble}_{ssp}_sfc_type_{sfc_type}_{mb_type}_{grad_type}_historical_test'\n",
    "climate_filename = 'climate_historical'\n",
    "climate_input_filesuffix = 'W5E5' #daily\n",
    "out_hist = compile_fixed_geometry_mass_balance_TIModel(gdirs, filesuffix='historical_W5E5_test',\n",
    "                                            climate_filename = climate_filename,\n",
    "                                            path=True, csv=True,\n",
    "                                            use_inversion_flowlines=True,\n",
    "                                            climate_input_filesuffix = climate_input_filesuffix,\n",
    "                                            ys=2000, ye=2019, \n",
    "                                            from_json=True,\n",
    "                                            json_filename='melt_f_geod',\n",
    "                                            sfc_type=melt_f_change,\n",
    "                                            **kwargs)\n",
    "# this should be again equal to the observations (as we calibrated it to match)\n",
    "np.testing.assert_allclose(out_hist.mean(), pd_geodetic.loc[gdir.rgi_id].dmdtda*1000, rtol=1e-3)\n",
    "#np.testing.assert_allclose(out['RGI60-11.00897'].mean(), -1100.3, rtol = 1e-3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# process future climate scenarios (here just one scenario and one gcm!)\n",
    "ensemble = 'mri-esm2-0_r1i1p1f1'\n",
    "ssp = 'ssp126'\n",
    "from MBsandbox.wip.projections_bayescalibration import process_isimip_data\n",
    "workflow.execute_entity_task(process_isimip_data, gdirs, ensemble = ensemble,\n",
    "                                 ssp = 'ssp126', temporal_resol ='daily',\n",
    "                                  climate_historical_filesuffix='_{}_{}'.format('daily', baseline_climate), correct=False)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "climate_filename='gcm_data'\n",
    "climate_input_filesuffix=f'ISIMIP3b_{ensemble}_{ssp}_no_correction'\n",
    "out_gcm = compile_fixed_geometry_mass_balance_TIModel(gdirs, filesuffix='gcm_test',\n",
    "                                            climate_filename = climate_filename,\n",
    "                                            path=True, csv=True,\n",
    "                                            use_inversion_flowlines=True,\n",
    "                                            climate_input_filesuffix = climate_input_filesuffix,\n",
    "                                            ys=2000, ye=2100, \n",
    "                                            from_json=True,\n",
    "                                            json_filename='melt_f_geod',\n",
    "                                            sfc_type=melt_f_change,\n",
    "                                            **kwargs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(out_gcm)\n",
    "plt.ylabel('fixed geometry specific MB')\n",
    "plt.xlabel('years')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### volume changes starting from rgi_date until end of calibration time period or until the end of the century:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "gdir.rgi_date"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "kwargs_for_TIModel_Sfc_Type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y0=2004 # but actually rgi date of Aneto glacier is in 2011 -> so it is just a very short time period ... \n",
    "ye_h = ye-1   \n",
    "from MBsandbox.flowline_TIModel import run_from_climate_data_TIModel\n",
    "workflow.execute_entity_task(run_from_climate_data_TIModel, gdirs, bias=0, #will actually point to the residual, should always be zero! \n",
    "                                  mb_model_sub_class=TIModel_Sfc_Type, # we use the temperature-index model variant with surface type distinction!\n",
    "                                  min_ys=y0, ye=ye_h, # starting and end year of the volume run \n",
    "                                  mb_type=mb_type,\n",
    "                                  grad_type=grad_type,\n",
    "                                  precipitation_factor=fit_prcp_fac_aneto,  # take the fitted precipitation factor\n",
    "                                  melt_f=melt_f_opt_0_5_m, # set to the calibrated melt_f (that fits to fit_prcp_fac_aneto)\n",
    "                                  climate_input_filesuffix=baseline_climate, # we use here the observational W5E5 climate dataset\n",
    "                                  output_filesuffix='_historical_run',# can add here more options to distinguish between runs\n",
    "                             store_model_geometry =True,\n",
    "                             kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type,\n",
    "                                 )\n",
    "ds_vol = utils.compile_run_output(gdirs, input_filesuffix='_historical_run')\n",
    "plt.figure(figsize=(8,6))\n",
    "ds_vol.sel(rgi_id=df[-1]).volume.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "ok, we just have a very first look at how the volume evolves under the lowest emission scenario for just one gcm  (with the chosen MB model):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "workflow.execute_entity_task(run_from_climate_data_TIModel, gdirs, bias=0, #will actually point to the residual, should always be zero! \n",
    "                                  mb_model_sub_class=TIModel_Sfc_Type, # we use the temperature-index model variant with surface type distinction!\n",
    "                            #min_ys=y0,\n",
    "                             ye=2100, # starting and end year of the volume run \n",
    "                                  mb_type=mb_type,\n",
    "                                  grad_type=grad_type,\n",
    "                                  precipitation_factor=fit_prcp_fac_aneto,  # take the fitted precipitation factor\n",
    "                                  melt_f=melt_f_opt_0_5_m, # set to the calibrated melt_f (that fits to fit_prcp_fac_aneto)\n",
    "                             init_model_filesuffix = '_historical_run',  # start from the stage of the end of the historical run \n",
    "                             #climate_type = 'gcm_data',\n",
    "                             climate_filename='gcm_data',\n",
    "                                  climate_input_filesuffix=f'ISIMIP3b_{ensemble}_{ssp}_no_correction', \n",
    "                                  output_filesuffix=f'_gcm_run_{ensemble}_{ssp}', # can add here more options to distinguish between runs\n",
    "                             store_model_geometry = True,\n",
    "                                kwargs_for_TIModel_Sfc_Type=kwargs_for_TIModel_Sfc_Type,\n",
    "                                 )\n",
    "ds_vol_future = utils.compile_run_output(gdirs, input_filesuffix=f'_gcm_run_{ensemble}_{ssp}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(12,6))\n",
    "ds_vol_future.sel(rgi_id=df[-1]).volume.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
